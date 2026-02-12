#![allow(non_snake_case)]

use anyhow::{anyhow, Context};
use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BuildVariant {
    Dev,
    Release,
    Production,
}

impl BuildVariant {
    fn suffix(self) -> &'static str {
        match self {
            BuildVariant::Dev => "dev",
            BuildVariant::Release => "release",
            BuildVariant::Production => "production",
        }
    }

    fn display_name(self) -> &'static str {
        match self {
            BuildVariant::Dev => "Dev",
            BuildVariant::Release => "Release",
            BuildVariant::Production => "Production",
        }
    }

    fn needs_release_flag(self) -> bool {
        matches!(self, BuildVariant::Release | BuildVariant::Production)
    }

    fn is_production(self) -> bool {
        matches!(self, BuildVariant::Production)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum XtaskMode {
    Bundle,
    Installer,
    Passthrough,
}

#[derive(Clone, Debug)]
struct InstallerArtifact {
    name: String,
    path: PathBuf,
    is_dir: bool,
}

#[derive(Clone, Debug)]
struct InstallerPayload {
    bundle_label: String,
    installer_root: PathBuf,
    vst3_artifacts: Vec<InstallerArtifact>,
    clap_artifacts: Vec<InstallerArtifact>,
    preset_artifacts: Vec<InstallerArtifact>,
}

fn detect_xtask_mode(args: &[String]) -> XtaskMode {
    if args.iter().any(|arg| arg == "installer") {
        XtaskMode::Installer
    } else if args.iter().any(|arg| arg == "bundle") {
        XtaskMode::Bundle
    } else {
        XtaskMode::Passthrough
    }
}

fn parse_build_variant(args: &[String]) -> anyhow::Result<BuildVariant> {
    let has_dev = args.iter().any(|arg| arg == "--dev");
    let has_release = args.iter().any(|arg| arg == "--release");
    let has_production = args.iter().any(|arg| arg == "--production");

    if has_production {
        if has_dev {
            return Err(anyhow!("--dev and --production cannot be used together"));
        }
        return Ok(BuildVariant::Production);
    }

    if has_dev {
        if has_release {
            return Err(anyhow!("--dev and --release cannot be used together"));
        }
        return Ok(BuildVariant::Dev);
    }

    if has_release {
        Ok(BuildVariant::Release)
    } else {
        Ok(BuildVariant::Dev)
    }
}

fn get_platform_bundle_dir() -> &'static str {
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    return "Windows_X86-64";

    #[cfg(all(target_os = "windows", target_arch = "x86"))]
    return "Windows_X86";

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    return "MacOS_AArch64";

    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    return "MacOS_X86-64";

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    return "Linux_X86-64";

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    return "Linux_AArch64";

    #[allow(unreachable_code)]
    "Unknown"
}

fn format_archive_dir_name(platform_label: &str, variant: BuildVariant) -> String {
    format!(
        "IRMax_{}_v{}-{}",
        platform_label,
        get_build_version(),
        variant.display_name()
    )
}

fn get_build_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

fn clean_default_bundled_dir(workspace_root: &Path) -> anyhow::Result<()> {
    let bundled_dir = workspace_root.join("target/bundled");
    if bundled_dir.exists() {
        println!("[Pre-Clean] Removing target/bundled to avoid platform conflicts...");
        fs::remove_dir_all(&bundled_dir).context("Failed to remove target/bundled directory")?;
    }
    Ok(())
}

#[cfg(target_os = "windows")]
fn find_ipp_bin_dir() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Ok(root) = env::var("IPP_ROOT") {
        candidates.push(PathBuf::from(root).join("bin"));
    }
    if let Ok(root) = env::var("ONEAPI_ROOT") {
        candidates.push(PathBuf::from(&root).join("ipp").join("latest").join("bin"));
        candidates.push(PathBuf::from(&root).join("ipp").join("2022.3").join("bin"));
    }

    candidates.push(PathBuf::from(
        r"C:\Program Files (x86)\Intel\oneAPI\ipp\latest\bin",
    ));
    candidates.push(PathBuf::from(
        r"C:\Program Files (x86)\Intel\oneAPI\ipp\2022.3\bin",
    ));

    for path in candidates {
        if path.exists() {
            return Some(path);
        }
    }
    None
}

#[cfg(target_os = "windows")]
fn copy_ipp_runtime_to_bundle(bundle_path: &Path) -> anyhow::Result<()> {
    let Some(bin_dir) = find_ipp_bin_dir() else {
        println!("[IPP] Warning: IPP bin dir not found. DLLs not bundled.");
        return Ok(());
    };

    let dest_dir = bundle_path.join("Contents").join("x86_64-win");
    if !dest_dir.exists() {
        println!(
            "[IPP] Warning: VST3 contents not found at {}",
            dest_dir.display()
        );
        return Ok(());
    }

    let prefixes = ["ippcore", "ipps", "ippvm"];
    let mut copied = 0usize;

    for entry in fs::read_dir(&bin_dir).context("Failed to read IPP bin directory")? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if !file_type.is_file() {
            continue;
        }

        let name = entry.file_name();
        let name_str = name.to_string_lossy().to_lowercase();
        if !name_str.ends_with(".dll") {
            continue;
        }
        if !prefixes.iter().any(|prefix| name_str.starts_with(prefix)) {
            continue;
        }

        let src = entry.path();
        let dst = dest_dir.join(name);
        fs::copy(&src, &dst)
            .with_context(|| format!("Failed to copy IPP runtime DLL from {}", src.display()))?;
        copied += 1;
    }

    if copied > 0 {
        println!(
            "[IPP] Bundled {} IPP DLL(s) into {}",
            copied,
            dest_dir.display()
        );
    } else {
        println!(
            "[IPP] Warning: No IPP DLLs copied from {}",
            bin_dir.display()
        );
    }

    Ok(())
}

fn move_to_platform_dir_with_label(
    workspace_root: &Path,
    variant: BuildVariant,
    platform_label: &str,
) -> anyhow::Result<PathBuf> {
    let src_dir = workspace_root.join("target/bundled");
    let dst_dir = workspace_root.join("target").join(format!(
        "bundled_{}-{}",
        platform_label,
        variant.suffix()
    ));

    if !src_dir.exists() {
        return Err(anyhow!("target/bundled not found after build"));
    }

    if dst_dir.exists() {
        fs::remove_dir_all(&dst_dir)?;
    }
    fs::create_dir_all(&dst_dir)?;

    for entry in fs::read_dir(&src_dir)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst_dir.join(entry.file_name());

        if dst_path.exists() {
            if dst_path.is_dir() {
                fs::remove_dir_all(&dst_path)?;
            } else {
                fs::remove_file(&dst_path)?;
            }
        }

        fs::rename(&src_path, &dst_path)
            .with_context(|| format!("Failed to move {:?} to {:?}", src_path, dst_path))?;
    }

    let _ = fs::remove_dir(&src_dir);
    println!("[Platform Bundle] Moved to: {}", dst_dir.display());
    Ok(dst_dir)
}

fn move_to_platform_dir(workspace_root: &Path, variant: BuildVariant) -> anyhow::Result<PathBuf> {
    move_to_platform_dir_with_label(workspace_root, variant, get_platform_bundle_dir())
}

fn prepare_bundle_args(
    raw_args: &[String],
    should_copy: bool,
    variant: BuildVariant,
) -> Vec<String> {
    let mut forwarded: Vec<String> = raw_args
        .iter()
        .skip(1)
        .filter(|arg| {
            *arg != "--dev" && *arg != "--production" && *arg != "installer" && *arg != "--no-build"
        })
        .cloned()
        .collect();

    if should_copy {
        if forwarded.is_empty() || forwarded[0].starts_with('-') {
            forwarded.insert(0, "bundle".to_string());
        }

        if !forwarded.iter().any(|arg| arg == "IRMax_Plugin") {
            if forwarded
                .first()
                .map(|cmd| cmd == "bundle")
                .unwrap_or(false)
            {
                forwarded.insert(1, "IRMax_Plugin".to_string());
            } else {
                forwarded.insert(0, "IRMax_Plugin".to_string());
            }
        }
    }

    if variant.needs_release_flag() && !forwarded.iter().any(|arg| arg == "--release") {
        forwarded.push("--release".to_string());
    }

    if variant.is_production() {
        let has_production_feature = forwarded
            .windows(2)
            .any(|pair| pair[0] == "--features" && pair[1].contains("IRMax_Plugin/production"));

        if !has_production_feature {
            forwarded.push("--features".to_string());
            forwarded.push("IRMax_Plugin/production".to_string());
        }
    }

    forwarded
}

fn find_first_bundle_with_ext(bundled_dir: &Path, ext: &str) -> anyhow::Result<PathBuf> {
    let entry = fs::read_dir(bundled_dir)
        .context("Failed to read bundled directory")?
        .filter_map(|entry| entry.ok())
        .find(|entry| entry.path().extension().map_or(false, |e| e == ext))
        .with_context(|| format!("No .{} bundle found", ext))?;

    Ok(entry.path())
}

#[cfg(target_os = "macos")]
fn find_optional_bundle_with_ext(bundled_dir: &Path, ext: &str) -> Option<PathBuf> {
    fs::read_dir(bundled_dir)
        .ok()?
        .filter_map(|entry| entry.ok())
        .find(|entry| entry.path().extension().map_or(false, |e| e == ext))
        .map(|entry| entry.path())
}

fn find_first_vst3_bundle(bundled_dir: &Path) -> anyhow::Result<PathBuf> {
    find_first_bundle_with_ext(bundled_dir, "vst3")
}

fn post_build_copy_mode(
    variant: BuildVariant,
    bundled_dir: &Path,
    target_label_override: Option<&str>,
) -> anyhow::Result<()> {
    println!(
        "\n[Auto-Copy] Starting post-build copy for {} mode...",
        variant.display_name()
    );
    let workspace_root = env::current_dir()?;

    if !bundled_dir.exists() {
        return Err(anyhow!("bundled directory not found"));
    }

    let src_path = find_first_vst3_bundle(bundled_dir)?;
    let dir_name = src_path.file_name().context("Invalid source filename")?;

    let archive_target_label = target_label_override.unwrap_or(get_platform_bundle_dir());
    let archive_dir_name = format_archive_dir_name(archive_target_label, variant);
    let internal_build_dir = workspace_root.join("Build").join(&archive_dir_name);
    if internal_build_dir.exists() {
        fs::remove_dir_all(&internal_build_dir)?;
    }
    fs::create_dir_all(&internal_build_dir)?;

    let archived_bundle_path = internal_build_dir.join(dir_name);
    copy_dir_recursive(&src_path, &archived_bundle_path)?;
    println!(
        "[Auto-Copy] Archived to: {}",
        archived_bundle_path.display()
    );
    #[cfg(target_os = "windows")]
    {
        let _ = copy_ipp_runtime_to_bundle(&archived_bundle_path);
    }

    #[cfg(target_os = "windows")]
    {
        let deploy_targets: Vec<(PathBuf, bool)> = match variant {
            BuildVariant::Dev => vec![(Path::new(r"C:\Plugins\VST Dev").to_path_buf(), false)],
            BuildVariant::Release => vec![
                (Path::new(r"C:\Plugins\VST Dev").to_path_buf(), false),
                (
                    Path::new(r"C:\Program Files\Common Files\VST3").to_path_buf(),
                    true,
                ),
            ],
            BuildVariant::Production => {
                vec![(
                    Path::new(r"C:\Program Files\Common Files\VST3").to_path_buf(),
                    true,
                )]
            }
        };

        for (deploy_dir, best_effort) in deploy_targets {
            if !deploy_dir.exists() {
                if best_effort {
                    let _ = fs::create_dir_all(&deploy_dir);
                } else {
                    fs::create_dir_all(&deploy_dir)?;
                }
            }

            let deployed_bundle_path = deploy_dir.join(dir_name);
            let copy_result = copy_dir_recursive(&src_path, &deployed_bundle_path);
            if best_effort {
                if let Err(error) = copy_result {
                    println!(
                        "[Auto-Copy] Warning: Failed to deploy to {} (permission issue?): {:?}",
                        deploy_dir.display(),
                        error
                    );
                    continue;
                }
            } else {
                copy_result?;
            }

            println!(
                "[Auto-Copy] Deployed to: {}",
                deployed_bundle_path.display()
            );
            let _ = copy_ipp_runtime_to_bundle(&deployed_bundle_path);
        }

        println!("[Auto-Copy] {} mode complete!\n", variant.display_name());
        Ok(())
    }

    #[cfg(target_os = "macos")]
    {
        let deploy_dirs: Vec<PathBuf> = if matches!(variant, BuildVariant::Dev) {
            vec![Path::new("/Plugins/VST Dev").to_path_buf()]
        } else {
            let home = env::var("HOME").context("HOME is not set")?;
            vec![
                Path::new("/Library/Audio/Plug-Ins/VST").to_path_buf(),
                Path::new("/Library/Audio/Plug-Ins/VST3").to_path_buf(),
                Path::new(&home).join("Library/Audio/Plug-Ins/VST"),
                Path::new(&home).join("Library/Audio/Plug-Ins/VST3"),
            ]
        };

        if matches!(variant, BuildVariant::Dev) {
            let system_bundle = Path::new("/Library/Audio/Plug-Ins/VST3").join(dir_name);
            if system_bundle.exists() {
                println!(
                    "[Auto-Copy] Notice: System bundle also exists at {}. DAW may prefer it over the dev bundle path.",
                    system_bundle.display()
                );
            }
        }

        for deploy_dir in deploy_dirs {
            fs::create_dir_all(&deploy_dir).with_context(|| {
                format!(
                    "Failed to create deploy directory: {}",
                    deploy_dir.display()
                )
            })?;

            let deployed_bundle_path = deploy_dir.join(dir_name);
            copy_dir_recursive(&src_path, &deployed_bundle_path).with_context(|| {
                format!(
                    "Failed to deploy bundle to {}",
                    deployed_bundle_path.display()
                )
            })?;
            println!(
                "[Auto-Copy] Deployed to: {}",
                deployed_bundle_path.display()
            );
            sign_vst3_bundle_macos(&deployed_bundle_path)?;
        }

        println!("[Auto-Copy] {} mode complete!\n", variant.display_name());
        Ok(())
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        println!("[Auto-Copy] {} mode complete!\n", variant.display_name());
        Ok(())
    }
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> anyhow::Result<()> {
    if dst.exists() {
        fs::remove_dir_all(dst)?;
    }
    fs::create_dir_all(dst)?;

    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let dest_path = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_recursive(&entry.path(), &dest_path)?;
        } else {
            fs::copy(entry.path(), &dest_path)?;
        }
    }

    Ok(())
}

fn copy_path(src: &Path, dst: &Path) -> anyhow::Result<()> {
    if src.is_dir() {
        copy_dir_recursive(src, dst)
    } else {
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(src, dst).with_context(|| {
            format!(
                "Failed to copy file from {} to {}",
                src.display(),
                dst.display()
            )
        })?;
        Ok(())
    }
}

fn sorted_dir_entries(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut entries = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("Failed to read {}", dir.display()))? {
        let entry = entry?;
        entries.push(entry.path());
    }

    entries.sort_by(|a, b| {
        let a_name = a
            .file_name()
            .map(|n| n.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        let b_name = b
            .file_name()
            .map(|n| n.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        a_name.cmp(&b_name)
    });

    Ok(entries)
}

fn push_staged_artifact(list: &mut Vec<InstallerArtifact>, staged_path: PathBuf, is_dir: bool) {
    let name = staged_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    list.push(InstallerArtifact {
        name,
        path: staged_path,
        is_dir,
    });
}

fn installer_preset_source_dir(workspace_root: &Path) -> PathBuf {
    let primary = workspace_root
        .join("Source")
        .join("Resources")
        .join("Presets");

    if primary.exists() {
        return primary;
    }

    // Backward compatibility for older local layouts.
    let legacy = workspace_root
        .join("Source")
        .join("Resources")
        .join("InstallerPayload")
        .join("Presets");

    if legacy.exists() {
        return legacy;
    }

    primary
}

fn should_skip_preset_payload_entry(path: &Path) -> bool {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();

    matches!(
        name.as_str(),
        "readme.txt" | "readme.md" | ".gitkeep" | ".keep"
    )
}

fn parse_bundle_label(bundle_dir: &Path, variant: BuildVariant) -> String {
    let file_name = bundle_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| format!("{}-{}", get_platform_bundle_dir(), variant.suffix()));

    let prefix = "bundled_";
    let suffix = format!("-{}", variant.suffix());
    if let Some(rest) = file_name.strip_prefix(prefix) {
        if let Some(label) = rest.strip_suffix(&suffix) {
            return label.to_string();
        }
    }

    get_platform_bundle_dir().to_string()
}

fn find_existing_bundle_dir(
    workspace_root: &Path,
    variant: BuildVariant,
) -> anyhow::Result<PathBuf> {
    let target_dir = workspace_root.join("target");
    let expected_suffix = format!("-{}", variant.suffix());
    let mut candidates: Vec<PathBuf> = Vec::new();

    for entry in fs::read_dir(&target_dir).with_context(|| {
        format!(
            "Failed to read target directory while locating existing bundle: {}",
            target_dir.display()
        )
    })? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("bundled_") && name.ends_with(&expected_suffix) {
            candidates.push(entry.path());
        }
    }

    if candidates.is_empty() {
        return Err(anyhow!(
            "No bundled output found for variant '{}'. Build first with `cargo run -p xtask -- bundle --{}` or run installer without --no-build.",
            variant.display_name(),
            variant.suffix()
        ));
    }

    candidates.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    #[cfg(target_os = "macos")]
    {
        if let Some(found) = candidates
            .iter()
            .find(|path| path.to_string_lossy().contains("MacOS_Universal"))
        {
            return Ok(found.clone());
        }
    }

    if let Some(found) = candidates
        .iter()
        .find(|path| path.to_string_lossy().contains(get_platform_bundle_dir()))
    {
        return Ok(found.clone());
    }

    let available = candidates
        .iter()
        .filter_map(|path| path.file_name().map(|n| n.to_string_lossy().to_string()))
        .collect::<Vec<_>>()
        .join(", ");

    Err(anyhow!(
        "No bundled output found for current platform '{}' and variant '{}'. Available bundles: {}",
        get_platform_bundle_dir(),
        variant.display_name(),
        available
    ))
}

fn prepare_installer_payload(
    workspace_root: &Path,
    bundle_dir: &Path,
    variant: BuildVariant,
) -> anyhow::Result<InstallerPayload> {
    let bundle_label = parse_bundle_label(bundle_dir, variant);
    let installer_root = workspace_root
        .join("target")
        .join("installer")
        .join(format!("{}-{}", bundle_label, variant.suffix()));

    if installer_root.exists() {
        fs::remove_dir_all(&installer_root)?;
    }

    let payload_root = installer_root.join("payload");
    let vst3_root = payload_root.join("plugins").join("vst3");
    let clap_root = payload_root.join("plugins").join("clap");
    let preset_root = payload_root.join("presets");

    fs::create_dir_all(&vst3_root)?;
    fs::create_dir_all(&clap_root)?;
    fs::create_dir_all(&preset_root)?;

    let mut vst3_artifacts = Vec::new();
    let mut clap_artifacts = Vec::new();

    for entry in sorted_dir_entries(bundle_dir)? {
        let ext = entry
            .extension()
            .and_then(OsStr::to_str)
            .unwrap_or_default();
        let file_name = entry
            .file_name()
            .ok_or_else(|| anyhow!("Invalid bundle entry in {}", bundle_dir.display()))?;

        if ext.eq_ignore_ascii_case("vst3") {
            let dst = vst3_root.join(file_name);
            let is_dir = entry.is_dir();
            copy_path(&entry, &dst)?;
            push_staged_artifact(&mut vst3_artifacts, dst, is_dir);
        } else if ext.eq_ignore_ascii_case("clap") {
            let dst = clap_root.join(file_name);
            let is_dir = entry.is_dir();
            copy_path(&entry, &dst)?;
            push_staged_artifact(&mut clap_artifacts, dst, is_dir);
        }
    }

    if vst3_artifacts.is_empty() && clap_artifacts.is_empty() {
        return Err(anyhow!(
            "No .vst3 or .clap artifacts found in {}",
            bundle_dir.display()
        ));
    }

    let mut preset_artifacts = Vec::new();
    let preset_source = installer_preset_source_dir(workspace_root);
    if preset_source.exists() {
        for entry in sorted_dir_entries(&preset_source)? {
            if should_skip_preset_payload_entry(&entry) {
                continue;
            }

            let file_name = entry
                .file_name()
                .ok_or_else(|| anyhow!("Invalid preset payload entry: {}", entry.display()))?;
            let dst = preset_root.join(file_name);
            let is_dir = entry.is_dir();
            copy_path(&entry, &dst)?;
            push_staged_artifact(&mut preset_artifacts, dst, is_dir);
        }

        if preset_artifacts.is_empty() {
            println!(
                "[Installer] Preset payload directory exists but is empty: {}",
                preset_source.display()
            );
        } else {
            println!(
                "[Installer] Staged {} preset payload item(s) from {}",
                preset_artifacts.len(),
                preset_source.display()
            );
        }
    } else {
        println!(
            "[Installer] Preset payload not found at {} (skip). Add files there to include factory presets.",
            preset_source.display()
        );
    }

    Ok(InstallerPayload {
        bundle_label,
        installer_root,
        vst3_artifacts,
        clap_artifacts,
        preset_artifacts,
    })
}

fn run_bundle_build(
    raw_args: &[String],
    variant: BuildVariant,
    workspace_root: &Path,
    deploy_after_build: bool,
) -> anyhow::Result<PathBuf> {
    clean_default_bundled_dir(workspace_root)?;
    apply_platform_build_env();

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if variant.is_production() {
        return run_local_production_universal(
            raw_args,
            variant,
            workspace_root,
            deploy_after_build,
        );
    }

    let forwarded_args = prepare_bundle_args(raw_args, true, variant);
    nih_plug_xtask::main_with_args("xtask", forwarded_args.into_iter())?;

    std::thread::sleep(std::time::Duration::from_millis(500));
    let platform_dir = move_to_platform_dir(workspace_root, variant)?;

    if deploy_after_build {
        post_build_copy_mode(variant, &platform_dir, None)?;
    }

    Ok(platform_dir)
}

#[cfg(target_os = "windows")]
fn nsis_path(path: &Path) -> String {
    path.to_string_lossy().replace('/', "\\").replace('$', "$$")
}

#[cfg(target_os = "windows")]
fn nsis_install_line(artifact: &InstallerArtifact) -> String {
    if artifact.is_dir {
        format!("  File /r \"{}\"", nsis_path(&artifact.path))
    } else {
        format!("  File \"{}\"", nsis_path(&artifact.path))
    }
}

#[cfg(target_os = "windows")]
fn nsis_uninstall_line(base_dir: &str, artifact: &InstallerArtifact) -> String {
    if artifact.is_dir {
        format!("  RMDir /r \"{}\\{}\"", base_dir, artifact.name)
    } else {
        format!("  Delete \"{}\\{}\"", base_dir, artifact.name)
    }
}

#[cfg(target_os = "windows")]
fn find_makensis() -> Option<PathBuf> {
    if let Ok(custom) = env::var("IRMAX_NSIS") {
        let custom_path = PathBuf::from(custom);
        if custom_path.exists() {
            return Some(custom_path);
        }
    }

    let standard = PathBuf::from(r"C:\Program Files (x86)\NSIS\makensis.exe");
    if standard.exists() {
        return Some(standard);
    }

    let output = Command::new("where").arg("makensis").output().ok()?;
    if !output.status.success() {
        return None;
    }

    let output_text = String::from_utf8_lossy(&output.stdout).to_string();
    let first = output_text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())?
        .to_string();

    Some(PathBuf::from(first))
}

#[cfg(target_os = "windows")]
fn command_exists_windows(command: &str) -> bool {
    Command::new("where")
        .arg(command)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

#[cfg(target_os = "windows")]
fn try_bootstrap_nsis_with(command: &str, args: &[&str]) -> bool {
    if !command_exists_windows(command) {
        return false;
    }

    println!(
        "[Installer] Trying automatic NSIS bootstrap: {} {}",
        command,
        args.join(" ")
    );

    match Command::new(command).args(args).status() {
        Ok(status) if status.success() => true,
        Ok(status) => {
            println!(
                "[Installer] {} finished with non-zero status {:?} while installing NSIS.",
                command,
                status.code()
            );
            false
        }
        Err(error) => {
            println!(
                "[Installer] Failed to launch {} while installing NSIS: {}",
                command, error
            );
            false
        }
    }
}

#[cfg(target_os = "windows")]
fn ensure_makensis_available() -> anyhow::Result<PathBuf> {
    if let Some(path) = find_makensis() {
        return Ok(path);
    }

    println!(
        "[Installer] NSIS makensis not found, attempting automatic install (winget -> choco -> scoop)..."
    );

    let _ = try_bootstrap_nsis_with(
        "winget",
        &[
            "install",
            "--id",
            "NSIS.NSIS",
            "-e",
            "--silent",
            "--accept-package-agreements",
            "--accept-source-agreements",
        ],
    );

    if let Some(path) = find_makensis() {
        return Ok(path);
    }

    let _ = try_bootstrap_nsis_with("choco", &["install", "nsis", "-y", "--no-progress"]);

    if let Some(path) = find_makensis() {
        return Ok(path);
    }

    let _ = try_bootstrap_nsis_with("scoop", &["install", "nsis"]);

    if let Some(path) = find_makensis() {
        return Ok(path);
    }

    Err(anyhow!(
        "NSIS makensis not found after auto-bootstrap attempts. Install NSIS once or set IRMAX_NSIS to makensis.exe path."
    ))
}

#[cfg(target_os = "windows")]
fn build_windows_installer(
    payload: &InstallerPayload,
    variant: BuildVariant,
    output_dir: &Path,
) -> anyhow::Result<PathBuf> {
    let script_path = payload.installer_root.join("irmax_installer.nsi");
    let output_file = output_dir.join(format!(
        "IRMax_{}_v{}-{}_Setup.exe",
        payload.bundle_label,
        get_build_version(),
        variant.display_name()
    ));

    if output_file.exists() {
        fs::remove_file(&output_file)?;
    }

    let mut install_steps: Vec<String> = Vec::new();
    let mut uninstall_steps: Vec<String> = Vec::new();

    if variant == BuildVariant::Dev {
        let dev_dir = r"C:\Plugins\VST Dev";
        if !payload.vst3_artifacts.is_empty() {
            install_steps.push(format!("  SetOutPath \"{}\"", dev_dir));
            for artifact in &payload.vst3_artifacts {
                install_steps.push(nsis_install_line(artifact));
                uninstall_steps.push(nsis_uninstall_line(dev_dir, artifact));
            }
        }
        if !payload.clap_artifacts.is_empty() {
            install_steps.push(format!("  SetOutPath \"{}\"", dev_dir));
            for artifact in &payload.clap_artifacts {
                install_steps.push(nsis_install_line(artifact));
                uninstall_steps.push(nsis_uninstall_line(dev_dir, artifact));
            }
        }
    } else {
        if !payload.vst3_artifacts.is_empty() {
            install_steps.push("  SetOutPath \"$COMMONFILES64\\VST3\"".to_string());
            for artifact in &payload.vst3_artifacts {
                install_steps.push(nsis_install_line(artifact));
                uninstall_steps.push(nsis_uninstall_line(r"$COMMONFILES64\VST3", artifact));
            }
        }
        if !payload.clap_artifacts.is_empty() {
            install_steps.push("  SetOutPath \"$COMMONFILES64\\CLAP\"".to_string());
            for artifact in &payload.clap_artifacts {
                install_steps.push(nsis_install_line(artifact));
                uninstall_steps.push(nsis_uninstall_line(r"$COMMONFILES64\CLAP", artifact));
            }
        }
    }

    if !payload.preset_artifacts.is_empty() {
        install_steps.push("  SetShellVarContext current".to_string());
        install_steps.push("  SetOutPath \"$LOCALAPPDATA\\IRMax\\Presets\"".to_string());
        for artifact in &payload.preset_artifacts {
            install_steps.push(nsis_install_line(artifact));
        }
    }

    let uninstall_name = format!("Uninstall IRMax {}.exe", variant.display_name());
    let uninstall_key = format!("IRMax_{}", variant.display_name());

    let script = format!(
        r#"Unicode true
RequestExecutionLevel admin
InstallDir "$PROGRAMFILES64\IRMax"
Name "IRMax {mode} Installer"
OutFile "{outfile}"

Section "Install"
  SetOutPath "$INSTDIR"
  WriteUninstaller "$INSTDIR\{uninstall_name}"
{install_steps}
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\{uninstall_key}" "DisplayName" "IRMax ({mode})"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\{uninstall_key}" "DisplayVersion" "{version}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\{uninstall_key}" "Publisher" "IRMax"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\{uninstall_key}" "UninstallString" "$\"$INSTDIR\{uninstall_name}$\""
SectionEnd

Section "Uninstall"
{uninstall_steps}
  Delete "$INSTDIR\{uninstall_name}"
  RMDir "$INSTDIR"
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\{uninstall_key}"
SectionEnd
"#,
        mode = variant.display_name(),
        outfile = nsis_path(&output_file),
        uninstall_name = uninstall_name,
        uninstall_key = uninstall_key,
        version = get_build_version(),
        install_steps = if install_steps.is_empty() {
            "  ; No payload staged".to_string()
        } else {
            install_steps.join("\n")
        },
        uninstall_steps = if uninstall_steps.is_empty() {
            "  ; No plugin artifacts to remove".to_string()
        } else {
            uninstall_steps.join("\n")
        },
    );

    fs::write(&script_path, script).with_context(|| {
        format!(
            "Failed to write NSIS installer script: {}",
            script_path.display()
        )
    })?;

    let makensis = ensure_makensis_available()?;

    let status = Command::new(&makensis)
        .arg("/V2")
        .arg(&script_path)
        .status()
        .with_context(|| format!("Failed to launch makensis: {}", makensis.display()))?;

    if !status.success() {
        return Err(anyhow!("makensis failed with status {:?}", status.code()));
    }

    if !output_file.exists() {
        return Err(anyhow!(
            "NSIS did not produce installer output at {}",
            output_file.display()
        ));
    }

    Ok(output_file)
}

#[cfg(target_os = "macos")]
fn build_macos_installer(
    payload: &InstallerPayload,
    variant: BuildVariant,
    output_dir: &Path,
) -> anyhow::Result<PathBuf> {
    use std::os::unix::fs::PermissionsExt;

    let pkgroot = payload.installer_root.join("pkgroot");
    let scripts_dir = payload.installer_root.join("scripts");

    if pkgroot.exists() {
        fs::remove_dir_all(&pkgroot)?;
    }
    if scripts_dir.exists() {
        fs::remove_dir_all(&scripts_dir)?;
    }

    fs::create_dir_all(&pkgroot)?;
    fs::create_dir_all(&scripts_dir)?;

    let (vst3_target_root, clap_target_root) = if variant == BuildVariant::Dev {
        (
            pkgroot.join("Plugins").join("VST Dev"),
            pkgroot.join("Plugins").join("VST Dev"),
        )
    } else {
        (
            pkgroot.join("Library/Audio/Plug-Ins/VST3"),
            pkgroot.join("Library/Audio/Plug-Ins/CLAP"),
        )
    };

    fs::create_dir_all(&vst3_target_root)?;
    fs::create_dir_all(&clap_target_root)?;

    for artifact in &payload.vst3_artifacts {
        copy_path(&artifact.path, &vst3_target_root.join(&artifact.name))?;
    }

    for artifact in &payload.clap_artifacts {
        copy_path(&artifact.path, &clap_target_root.join(&artifact.name))?;
    }

    let factory_root = pkgroot
        .join("Library")
        .join("Application Support")
        .join("IRMax")
        .join("FactoryPresets");

    if !payload.preset_artifacts.is_empty() {
        fs::create_dir_all(&factory_root)?;
        for artifact in &payload.preset_artifacts {
            copy_path(&artifact.path, &factory_root.join(&artifact.name))?;
        }
    }

    let postinstall_path = scripts_dir.join("postinstall");
    let postinstall = if payload.preset_artifacts.is_empty() {
        "#!/bin/bash\nexit 0\n".to_string()
    } else {
        r#"#!/bin/bash
set -euo pipefail

FACTORY_ROOT="/Library/Application Support/IRMax/FactoryPresets"
if [ ! -d "$FACTORY_ROOT" ]; then
  exit 0
fi

CONSOLE_USER=$(stat -f '%Su' /dev/console 2>/dev/null || true)
if [ -z "$CONSOLE_USER" ] || [ "$CONSOLE_USER" = "root" ]; then
  exit 0
fi

USER_HOME=$(dscl . -read /Users/$CONSOLE_USER NFSHomeDirectory 2>/dev/null | awk '{print $2}')
if [ -z "$USER_HOME" ]; then
  USER_HOME="/Users/$CONSOLE_USER"
fi

TARGET_ROOT="$USER_HOME/Library/Application Support/IRMax/Presets"
mkdir -p "$TARGET_ROOT"

if command -v rsync >/dev/null 2>&1; then
  rsync -a --ignore-existing "$FACTORY_ROOT/" "$TARGET_ROOT/"
else
  ditto "$FACTORY_ROOT" "$TARGET_ROOT"
fi

chown -R "$CONSOLE_USER":staff "$TARGET_ROOT"
exit 0
"#
        .to_string()
    };

    fs::write(&postinstall_path, postinstall)?;
    let mut permissions = fs::metadata(&postinstall_path)?.permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&postinstall_path, permissions)?;

    let unsigned_pkg = output_dir.join(format!(
        "IRMax_{}_v{}-{}.pkg",
        payload.bundle_label,
        get_build_version(),
        variant.display_name()
    ));

    if unsigned_pkg.exists() {
        fs::remove_file(&unsigned_pkg)?;
    }

    let pkgbuild_status = Command::new("pkgbuild")
        .arg("--root")
        .arg(&pkgroot)
        .arg("--scripts")
        .arg(&scripts_dir)
        .arg("--identifier")
        .arg(format!("com.irmax.plugin.{}", variant.suffix()))
        .arg("--version")
        .arg(get_build_version())
        .arg("--install-location")
        .arg("/")
        .arg(&unsigned_pkg)
        .status()
        .context("Failed to run pkgbuild")?;

    if !pkgbuild_status.success() {
        return Err(anyhow!(
            "pkgbuild failed with status {:?}",
            pkgbuild_status.code()
        ));
    }

    if let Ok(identity_raw) = env::var("IRMAX_MAC_INSTALLER_SIGN_IDENTITY") {
        let identity = identity_raw.trim();
        if !identity.is_empty() {
            let signed_pkg = output_dir.join(format!(
                "IRMax_{}_v{}-{}_signed.pkg",
                payload.bundle_label,
                get_build_version(),
                variant.display_name()
            ));
            if signed_pkg.exists() {
                fs::remove_file(&signed_pkg)?;
            }

            let status = Command::new("productsign")
                .arg("--sign")
                .arg(identity)
                .arg(&unsigned_pkg)
                .arg(&signed_pkg)
                .status()
                .context("Failed to run productsign")?;

            if !status.success() {
                return Err(anyhow!(
                    "productsign failed with status {:?}",
                    status.code()
                ));
            }

            fs::remove_file(&unsigned_pkg).ok();
            return Ok(signed_pkg);
        }
    }

    Ok(unsigned_pkg)
}

fn run_installer_mode(
    raw_args: &[String],
    variant: BuildVariant,
    workspace_root: &Path,
) -> anyhow::Result<()> {
    let no_build = raw_args.iter().any(|arg| arg == "--no-build");

    let bundle_dir = if no_build {
        println!("[Installer] --no-build enabled, locating existing bundle...");
        find_existing_bundle_dir(workspace_root, variant)?
    } else {
        println!("[Installer] Building plugin bundle before packaging installer...");
        run_bundle_build(raw_args, variant, workspace_root, false)?
    };

    println!("[Installer] Using bundle dir: {}", bundle_dir.display());

    let payload = prepare_installer_payload(workspace_root, &bundle_dir, variant)?;
    let output_dir = workspace_root
        .join("Build")
        .join("Installers")
        .join(variant.display_name());
    fs::create_dir_all(&output_dir)?;

    #[cfg(target_os = "windows")]
    {
        let output = build_windows_installer(&payload, variant, &output_dir)?;
        println!("[Installer] Windows package ready: {}", output.display());
        return Ok(());
    }

    #[cfg(target_os = "macos")]
    {
        let output = build_macos_installer(&payload, variant, &output_dir)?;
        println!("[Installer] macOS package ready: {}", output.display());
        return Ok(());
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        let _ = payload;
        let _ = output_dir;
        Err(anyhow!(
            "Installer packaging is currently supported only on Windows/macOS"
        ))
    }
}

#[cfg(target_os = "macos")]
fn sign_vst3_bundle_macos(bundle_path: &Path) -> anyhow::Result<()> {
    let status = Command::new("codesign")
        .args([
            "--force",
            "--deep",
            "--sign",
            "-",
            bundle_path.to_str().unwrap(),
        ])
        .status()?;

    if !status.success() {
        println!("Warning: ad-hoc signing failed.");
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn ensure_rust_target_installed(target: &str) -> anyhow::Result<()> {
    let list_output = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .with_context(|| format!("Failed to query installed rust targets for {}", target))?;

    if !list_output.status.success() {
        return Err(anyhow!(
            "rustup target list --installed failed while checking {}",
            target
        ));
    }

    let installed = String::from_utf8_lossy(&list_output.stdout);
    if installed.lines().any(|line| line.trim() == target) {
        return Ok(());
    }

    println!("[Universal] Installing rust target: {}", target);
    let status = Command::new("rustup")
        .args(["target", "add", target])
        .status()
        .with_context(|| format!("Failed to install rust target {}", target))?;

    if !status.success() {
        return Err(anyhow!("rustup target add {} failed", target));
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn with_target_arg(args: &[String], target: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(args.len() + 2);
    let mut i = 0usize;

    while i < args.len() {
        if args[i] == "--target" {
            i += 1;
            if i < args.len() {
                i += 1;
            }
            continue;
        }
        out.push(args[i].clone());
        i += 1;
    }

    out.push("--target".to_string());
    out.push(target.to_string());
    out
}

#[cfg(target_os = "macos")]
fn merge_macos_universal_bundle(
    workspace_root: &Path,
    variant: BuildVariant,
    arm_dir: &Path,
    x64_dir: &Path,
) -> anyhow::Result<PathBuf> {
    let universal_dir = workspace_root
        .join("target")
        .join(format!("bundled_MacOS_Universal-{}", variant.suffix()));

    if universal_dir.exists() {
        fs::remove_dir_all(&universal_dir)?;
    }
    fs::create_dir_all(&universal_dir)?;

    let arm_vst3 = find_first_bundle_with_ext(arm_dir, "vst3")?;
    let x64_vst3 = find_first_bundle_with_ext(x64_dir, "vst3")?;
    let vst3_name = arm_vst3
        .file_name()
        .context("Invalid arm64 vst3 bundle name")?;
    let universal_vst3 = universal_dir.join(vst3_name);
    copy_dir_recursive(&arm_vst3, &universal_vst3)?;

    let arm_vst3_bin = arm_vst3.join("Contents").join("MacOS").join("IRMax_Plugin");
    let x64_vst3_bin = x64_vst3.join("Contents").join("MacOS").join("IRMax_Plugin");
    let universal_vst3_bin = universal_vst3
        .join("Contents")
        .join("MacOS")
        .join("IRMax_Plugin");

    let lipo_status = Command::new("lipo")
        .arg("-create")
        .arg(&arm_vst3_bin)
        .arg(&x64_vst3_bin)
        .arg("-output")
        .arg(&universal_vst3_bin)
        .status()
        .context("Failed to run lipo for VST3 universal merge")?;

    if !lipo_status.success() {
        return Err(anyhow!("lipo failed while merging VST3 universal binary"));
    }

    sign_vst3_bundle_macos(&universal_vst3)?;

    let arm_clap = find_optional_bundle_with_ext(arm_dir, "clap");
    let x64_clap = find_optional_bundle_with_ext(x64_dir, "clap");
    if let (Some(arm_clap), Some(x64_clap)) = (arm_clap, x64_clap) {
        let clap_name = arm_clap
            .file_name()
            .context("Invalid arm64 clap bundle name")?;
        let universal_clap = universal_dir.join(clap_name);
        copy_dir_recursive(&arm_clap, &universal_clap)?;

        let arm_clap_bin = arm_clap.join("Contents").join("MacOS").join("IRMax_Plugin");
        let x64_clap_bin = x64_clap.join("Contents").join("MacOS").join("IRMax_Plugin");
        let universal_clap_bin = universal_clap
            .join("Contents")
            .join("MacOS")
            .join("IRMax_Plugin");

        let clap_lipo_status = Command::new("lipo")
            .arg("-create")
            .arg(&arm_clap_bin)
            .arg(&x64_clap_bin)
            .arg("-output")
            .arg(&universal_clap_bin)
            .status()
            .context("Failed to run lipo for CLAP universal merge")?;

        if !clap_lipo_status.success() {
            return Err(anyhow!("lipo failed while merging CLAP universal binary"));
        }

        sign_vst3_bundle_macos(&universal_clap)?;
    }

    println!(
        "[Universal] Created merged bundle at {}",
        universal_dir.display()
    );
    Ok(universal_dir)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn run_local_production_universal(
    raw_args: &[String],
    variant: BuildVariant,
    workspace_root: &Path,
    deploy_after_build: bool,
) -> anyhow::Result<PathBuf> {
    println!("[Universal] Production mode: building arm64 + x86_64 then merging.");

    ensure_rust_target_installed("aarch64-apple-darwin")?;
    ensure_rust_target_installed("x86_64-apple-darwin")?;

    clean_default_bundled_dir(workspace_root)?;
    let base_args = prepare_bundle_args(raw_args, true, variant);

    let arm_args = with_target_arg(&base_args, "aarch64-apple-darwin");
    nih_plug_xtask::main_with_args("xtask", arm_args.into_iter())?;
    std::thread::sleep(std::time::Duration::from_millis(500));
    let arm_dir = move_to_platform_dir_with_label(workspace_root, variant, "MacOS_AArch64")?;

    clean_default_bundled_dir(workspace_root)?;
    let x64_args = with_target_arg(&base_args, "x86_64-apple-darwin");
    nih_plug_xtask::main_with_args("xtask", x64_args.into_iter())?;
    std::thread::sleep(std::time::Duration::from_millis(500));
    let x64_dir = move_to_platform_dir_with_label(workspace_root, variant, "MacOS_X86-64")?;

    let universal_dir = merge_macos_universal_bundle(workspace_root, variant, &arm_dir, &x64_dir)?;
    if deploy_after_build {
        post_build_copy_mode(variant, &universal_dir, Some("MacOS_Universal"))?;
    }
    Ok(universal_dir)
}

fn apply_platform_build_env() {
    #[cfg(target_os = "macos")]
    {
        env::set_var("CARGO_INCREMENTAL", "0");
        println!("[Build Env] macOS: forced CARGO_INCREMENTAL=0");
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    let mode = detect_xtask_mode(&args);
    let variant = parse_build_variant(&args)?;
    let workspace_root = env::current_dir()?;

    println!("[Platform] Target: {}", get_platform_bundle_dir());
    println!("[Mode] Build variant: {}", variant.display_name());
    println!("[Version] Package version: {}", get_build_version());

    match mode {
        XtaskMode::Installer => {
            println!("[Command] installer");
            run_installer_mode(&args, variant, &workspace_root)
        }
        XtaskMode::Bundle => {
            println!("[Command] bundle");
            println!(
                "[Archive Layout] Build/{}/<bundle>",
                format_archive_dir_name(get_platform_bundle_dir(), variant)
            );
            let _platform_dir = run_bundle_build(&args, variant, &workspace_root, true)?;
            Ok(())
        }
        XtaskMode::Passthrough => {
            apply_platform_build_env();
            let forwarded_args = prepare_bundle_args(&args, false, variant);
            nih_plug_xtask::main_with_args("xtask", forwarded_args.into_iter())
        }
    }
}
