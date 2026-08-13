//! Per-session worktree orchestration for sandboxed sessions.
//!
//! A sandboxed session whose working directory is a git repository runs in a
//! throwaway worktree forked from HEAD rather than the user's live checkout.
//! This module owns that worktree across the session lifecycle: forking at
//! launch, restoring on resume, rolling back when launch fails, and cleaning
//! up when the session is deleted. The git operations themselves live in
//! `crate::workspace::worktree`; what lives here is the session-scoped
//! policy around them.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::{MountSpec, SandboxWorktree};
use crate::paths::PathContext;
use crate::workspace::worktree;

/// The result of forking a session worktree: what to mount where, plus the
/// cleanup metadata to persist with the session.
pub(crate) struct SessionWorktreeFork {
    /// Host directory to mount at the sandbox workdir (the worktree
    /// counterpart of the session cwd).
    pub host: PathBuf,
    pub worktree: SandboxWorktree,
    /// Identity mount for the repository's shared git dir: the worktree's
    /// `.git` file refers to it by absolute host path, so the container must
    /// see it at that same path for git to work inside.
    pub git_dir_mount: MountSpec,
}

/// Forks the repository containing `cwd` into a per-session worktree, so a
/// sandboxed session writes — and switches branches in — a throwaway checkout
/// rather than the user's live one.
///
/// Returns `None` — and the caller falls back to mounting the live directory —
/// when `cwd` is not in a git repository, the repository has no commits yet,
/// there is no nac home to hold the worktree, or worktree creation fails; the
/// fallback keeps sandboxing available where a worktree is impossible.
pub(crate) fn fork(cwd: &Path, session_key: &str) -> Option<SessionWorktreeFork> {
    let repo = match worktree::find_repo(cwd) {
        Ok(Some(repo)) => repo,
        Ok(None) => return None,
        Err(error) => {
            eprintln!("nac: cannot inspect git repository for sandbox worktree: {error:#}");
            return None;
        }
    };
    let Some(nac_home) = PathContext::new(cwd).nac_home_dir() else {
        eprintln!("nac: no nac home directory; sandbox will mount the live checkout");
        return None;
    };
    let worktree_path = nac_home.join("worktrees").join(session_key);
    let branch = format!("nac/{:.12}", session_key);
    if let Err(error) = worktree::create(&repo.root, &worktree_path, &branch) {
        eprintln!("nac: {error:#}; sandbox will mount the live checkout");
        return None;
    }
    // `cwd` may sit below the repository root; mount the same relative
    // position inside the worktree. A directory new enough to be absent from
    // HEAD is created empty.
    let relative = cwd
        .canonicalize()
        .ok()
        .and_then(|canonical| canonical.strip_prefix(&repo.root).ok().map(PathBuf::from))
        .unwrap_or_default();
    let host = worktree_path.join(&relative);
    if !host.exists() {
        if let Err(error) = std::fs::create_dir_all(&host) {
            eprintln!(
                "nac: failed to prepare session worktree directory '{}': {error}; \
                 sandbox will mount the live checkout",
                host.display()
            );
            let _ = worktree::remove(&repo.root, &worktree_path);
            let _ = worktree::delete_branch(&repo.root, &branch);
            return None;
        }
    }
    let git_dir_mount = MountSpec {
        host: repo.common_git_dir.clone(),
        guest: repo.common_git_dir,
        read_only: false,
    };
    Some(SessionWorktreeFork {
        host,
        worktree: SandboxWorktree {
            repo_root: repo.root,
            path: worktree_path,
            branch,
            fork_point: repo.head,
        },
        git_dir_mount,
    })
}

/// Brings a resumed session's worktree back into a usable state. A worktree
/// deleted or corrupted while the session was away is re-attached from its
/// branch; without the branch the session's workspace is unrecoverable and
/// podman would silently mount an empty directory in its place, so that case
/// fails the resume instead.
pub(crate) fn restore(worktree: &SandboxWorktree) -> Result<()> {
    if worktree::is_usable(&worktree.path) {
        return Ok(());
    }
    if worktree.path.exists() {
        let _ = worktree::remove(&worktree.repo_root, &worktree.path);
    }
    if worktree.path.exists() {
        // No longer a registered worktree, so git cannot clear it; the path
        // is session-owned scratch, safe to remove directly.
        if !worktree.path_in_scratch_dir() {
            anyhow::bail!(
                "stale sandbox worktree '{}' is outside the worktrees scratch directory; \
                 remove it manually to resume",
                worktree.path.display()
            );
        }
        std::fs::remove_dir_all(&worktree.path).with_context(|| {
            format!(
                "failed to clear stale sandbox worktree '{}'",
                worktree.path.display()
            )
        })?;
    }
    if worktree::branch_head(&worktree.repo_root, &worktree.branch).is_none() {
        anyhow::bail!(
            "sandbox worktree '{}' and its branch '{}' are both gone; \
             the session workspace cannot be restored",
            worktree.path.display(),
            worktree.branch
        );
    }
    worktree::re_add(&worktree.repo_root, &worktree.path, &worktree.branch)
}

/// Undoes a fork when session launch fails after it. The fork carries no
/// session work yet, so removal is always safe. Best-effort.
pub(crate) fn rollback(worktree: &SandboxWorktree) {
    let _ = worktree::remove(&worktree.repo_root, &worktree.path);
    let _ = worktree::delete_branch(&worktree.repo_root, &worktree.branch);
}

/// Removes the worktree a sandboxed session ran in. The session branch is
/// deleted only while it still points at the fork commit; a branch holding
/// session commits is kept and reported, because deleting it would discard
/// work the user may want. All best-effort: session deletion must not fail
/// because cleanup did.
pub fn cleanup_session_worktree(worktree: &SandboxWorktree) {
    if let Err(error) = worktree::remove(&worktree.repo_root, &worktree.path) {
        eprintln!("nac: failed to remove session worktree during deletion: {error:#}");
    }
    if worktree.path_in_scratch_dir() && worktree.path.exists() {
        let _ = std::fs::remove_dir_all(&worktree.path);
    }
    match worktree::branch_head(&worktree.repo_root, &worktree.branch) {
        Some(head) if head == worktree.fork_point => {
            if let Err(error) = worktree::delete_branch(&worktree.repo_root, &worktree.branch) {
                eprintln!("nac: failed to delete session branch during deletion: {error:#}");
            }
        }
        Some(_) => {
            eprintln!(
                "nac: session work remains on branch '{}' in '{}'",
                worktree.branch,
                worktree.repo_root.display()
            );
        }
        None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TEST_ENV_LOCK;

    struct TestRepo {
        base: PathBuf,
        repo: PathBuf,
        nac_home: PathBuf,
    }

    impl TestRepo {
        fn new(label: &str) -> Self {
            let unique = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let base = std::env::temp_dir().join(format!(
                "nac-session-worktree-{label}-{}-{unique}",
                std::process::id()
            ));
            let nac_home = base.join("nac-home");
            let repo = base.join("repo");
            std::fs::create_dir_all(&nac_home).unwrap();
            std::fs::create_dir_all(&repo).unwrap();
            git(&repo, &["init", "--quiet"]);
            git(&repo, &["config", "user.email", "nac@test"]);
            git(&repo, &["config", "user.name", "nac"]);
            Self {
                base,
                repo,
                nac_home,
            }
        }

        fn commit_file(&self, path: &str, contents: &str) {
            let file = self.repo.join(path);
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(file, contents).unwrap();
            git(&self.repo, &["add", path]);
            git(&self.repo, &["commit", "--quiet", "-m", path]);
        }

        fn forked_worktree(&self, key: &str) -> SandboxWorktree {
            let info = worktree::find_repo(&self.repo).unwrap().unwrap();
            let worktree = SandboxWorktree {
                repo_root: info.root.clone(),
                path: self.base.join("worktrees").join(key),
                branch: format!("nac/{key}"),
                fork_point: info.head,
            };
            worktree::create(&info.root, &worktree.path, &worktree.branch).unwrap();
            worktree
        }
    }

    impl Drop for TestRepo {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.base);
        }
    }

    fn git(cwd: &Path, args: &[&str]) {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(cwd)
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn git_branch(cwd: &Path) -> String {
        String::from_utf8_lossy(
            &std::process::Command::new("git")
                .arg("-C")
                .arg(cwd)
                .args(["rev-parse", "--abbrev-ref", "HEAD"])
                .output()
                .unwrap()
                .stdout,
        )
        .trim()
        .to_string()
    }

    #[test]
    fn fork_rewrites_the_mount_and_leaves_the_live_checkout_alone() {
        let _guard = TEST_ENV_LOCK.lock().unwrap();
        let original_nac_home = std::env::var_os("NAC_HOME");
        let original_xdg = std::env::var_os("XDG_CONFIG_HOME");
        let repo = TestRepo::new("fork");
        unsafe {
            std::env::set_var("NAC_HOME", &repo.nac_home);
            std::env::remove_var("XDG_CONFIG_HOME");
        }
        repo.commit_file("a.txt", "a");

        let forked = fork(&repo.repo, "sessionkey123").expect("git repo must get a worktree");
        assert_eq!(forked.host, forked.worktree.path);
        assert_eq!(
            forked.worktree.path,
            repo.nac_home.join("worktrees/sessionkey123")
        );
        assert_eq!(forked.worktree.branch, "nac/sessionkey12");
        assert_eq!(
            forked.git_dir_mount.host,
            repo.repo.join(".git").canonicalize().unwrap()
        );
        assert_eq!(forked.git_dir_mount.host, forked.git_dir_mount.guest);
        // The fork carries the committed tree but follows its own branch even
        // when the live checkout moves.
        assert!(forked.worktree.path.join("a.txt").exists());
        git(&repo.repo, &["checkout", "--quiet", "-b", "scratch"]);
        assert_eq!(git_branch(&repo.repo), "scratch");
        assert_eq!(git_branch(&forked.worktree.path), "nac/sessionkey12");

        // A plain directory gets no worktree: the live mount fallback applies.
        let plain = repo.base.join("plain");
        std::fs::create_dir_all(&plain).unwrap();
        assert!(fork(&plain, "otherkey456").is_none());

        // A cwd below the repo root mounts the same relative position inside
        // the fork.
        repo.commit_file("crates/child/c.txt", "c");
        let subdir = repo.repo.join("crates/child");
        let sub = fork(&subdir, "subdirkey789").expect("subdir cwd gets a worktree");
        assert_eq!(sub.host, sub.worktree.path.join("crates/child"));
        assert!(sub.host.join("c.txt").exists());
        rollback(&sub.worktree);
        rollback(&forked.worktree);

        unsafe {
            match original_nac_home {
                Some(value) => std::env::set_var("NAC_HOME", value),
                None => std::env::remove_var("NAC_HOME"),
            }
            match original_xdg {
                Some(value) => std::env::set_var("XDG_CONFIG_HOME", value),
                None => std::env::remove_var("XDG_CONFIG_HOME"),
            }
        }
    }

    #[test]
    fn cleanup_deletes_an_untouched_branch() {
        let repo = TestRepo::new("untouched");
        repo.commit_file("a.txt", "a");
        let worktree = repo.forked_worktree("cleanup-test");

        cleanup_session_worktree(&worktree);

        assert!(!worktree.path.exists());
        assert_eq!(
            worktree::branch_head(&worktree.repo_root, &worktree.branch),
            None,
            "a branch with no session commits must be deleted"
        );
    }

    #[test]
    fn cleanup_keeps_a_branch_holding_session_commits() {
        let repo = TestRepo::new("committed");
        repo.commit_file("a.txt", "a");
        let worktree = repo.forked_worktree("cleanup-test");
        std::fs::write(worktree.path.join("b.txt"), "b").unwrap();
        git(&worktree.path, &["add", "b.txt"]);
        git(&worktree.path, &["commit", "--quiet", "-m", "b"]);

        cleanup_session_worktree(&worktree);

        assert!(!worktree.path.exists());
        assert!(
            worktree::branch_head(&worktree.repo_root, &worktree.branch).is_some(),
            "a branch holding session commits must be kept"
        );
        worktree::delete_branch(&worktree.repo_root, &worktree.branch).unwrap();
    }
}
