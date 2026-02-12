use std::process::Command;

fn git_output(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8(output.stdout).ok()?;
    let trimmed = s.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn detect_git_branch() -> String {
    let branch =
        git_output(&["rev-parse", "--abbrev-ref", "HEAD"]).unwrap_or_else(|| "nogit".to_string());
    if branch == "HEAD" {
        return "detached".to_string();
    }
    branch
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/index");

    let git_sha =
        git_output(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "nogit".to_string());
    let git_branch = detect_git_branch();

    #[cfg(feature = "gui-slint")]
    {
        let config = slint_build::CompilerConfiguration::new().with_include_paths(vec![
            "ui".into(),
        ]);

        slint_build::compile_with_config("ui/main.slint", config)
            .expect("Failed to compile Slint UI for IRMax_Plugin");

        println!("cargo:rerun-if-changed=ui/main.slint");
        println!("cargo:rerun-if-changed=ui/pages/home_page.slint");
        println!("cargo:rerun-if-changed=ui/widgets/section_title.slint");
        println!("cargo:rerun-if-changed=ui/globals/theme.slint");
    }

    println!("cargo:rustc-env=IRMAX_GIT_SHA={git_sha}");
    println!("cargo:rustc-env=IRMAX_GIT_BRANCH={git_branch}");

    #[cfg(target_os = "windows")]
    {
        // Delay-load IPP DLLs so hosts with restricted DLL search can still load the plugin.
        println!("cargo:rustc-link-lib=delayimp");
        println!("cargo:rustc-link-arg=/DELAYLOAD:ippcore.dll");
        println!("cargo:rustc-link-arg=/DELAYLOAD:ipps.dll");
        println!("cargo:rustc-link-arg=/DELAYLOAD:ippvm.dll");
    }
}
