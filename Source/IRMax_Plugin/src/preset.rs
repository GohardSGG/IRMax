use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

pub const PRESET_PACK_FILE_NAME: &str = "_irmax_preset_pack_v1.json";

pub fn default_preset_root() -> PathBuf {
    if let Ok(raw) = std::env::var("IRMAX_PRESET_ROOT") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }

    if let Some(mut path) = dirs::data_local_dir() {
        path.push("IRMax");
        path.push("Presets");
        return path;
    }

    if let Some(mut path) = dirs::data_dir() {
        path.push("IRMax");
        path.push("Presets");
        return path;
    }

    PathBuf::from("IRMax_Presets")
}

pub fn ensure_preset_root_exists(path: &Path) -> Result<()> {
    fs::create_dir_all(path).with_context(|| format!("Failed to create preset root: {:?}", path))
}

#[derive(Debug, Clone)]
pub struct PresetTreeNode {
    pub name: String,
    pub rel_path: String,
    pub has_ir_group: bool,
    pub children: Vec<PresetTreeNode>,
}

impl PresetTreeNode {
    pub fn new_root() -> Self {
        Self {
            name: "ROOT".to_string(),
            rel_path: String::new(),
            has_ir_group: false,
            children: Vec::new(),
        }
    }

    pub fn contains_group(&self, leaf_id: &str) -> bool {
        let normalized = normalize_leaf_id(leaf_id);
        if self.has_ir_group && self.rel_path == normalized {
            return true;
        }
        self.children
            .iter()
            .any(|child| child.contains_group(&normalized))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetGroupParams {
    #[serde(default = "default_mix")]
    pub mix: f32,
    #[serde(default)]
    pub output_gain_db: f32,
    #[serde(default)]
    pub updated_at: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

fn default_mix() -> f32 {
    1.0
}

impl Default for PresetGroupParams {
    fn default() -> Self {
        Self {
            mix: 1.0,
            output_gain_db: 0.0,
            updated_at: String::new(),
            tags: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetPack {
    #[serde(default = "default_pack_version")]
    pub version: u32,
    #[serde(default)]
    pub groups: BTreeMap<String, PresetGroupParams>,
}

fn default_pack_version() -> u32 {
    1
}

impl Default for PresetPack {
    fn default() -> Self {
        Self {
            version: default_pack_version(),
            groups: BTreeMap::new(),
        }
    }
}

pub fn normalize_leaf_id(raw: &str) -> String {
    raw.replace('\\', "/")
        .split('/')
        .filter(|segment| !segment.is_empty() && *segment != ".")
        .collect::<Vec<_>>()
        .join("/")
}

fn sort_key(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default()
}

fn is_wav_file(path: &Path) -> bool {
    path.extension()
        .map(|ext| ext.to_string_lossy().eq_ignore_ascii_case("wav"))
        .unwrap_or(false)
}

pub fn contains_wav_files(path: &Path) -> bool {
    let entries = match fs::read_dir(path) {
        Ok(entries) => entries,
        Err(_) => return false,
    };

    entries.flatten().any(|entry| {
        let candidate = entry.path();
        candidate.is_file() && is_wav_file(&candidate)
    })
}

pub fn parse_sample_rate_folder_name(name: &str) -> Option<u32> {
    let compact: String = name
        .trim()
        .chars()
        .filter(|ch| !ch.is_whitespace() && *ch != '_' && *ch != '-')
        .collect();
    if compact.is_empty() {
        return None;
    }

    let lower = compact.to_ascii_lowercase();

    let (numeric, scale_khz) = if let Some(n) = lower.strip_suffix("khz") {
        (n, true)
    } else if let Some(n) = lower.strip_suffix('k') {
        (n, true)
    } else if let Some(n) = lower.strip_suffix("hz") {
        (n, false)
    } else {
        (lower.as_str(), false)
    };

    if numeric.is_empty() || !numeric.chars().all(|ch| ch.is_ascii_digit() || ch == '.') {
        return None;
    }

    let value = numeric.parse::<f32>().ok()?;
    if !value.is_finite() || value <= 0.0 {
        return None;
    }

    let hz = if scale_khz {
        value * 1000.0
    } else if lower.ends_with("hz") {
        value
    } else if numeric.contains('.') {
        value * 1000.0
    } else if value >= 1000.0 {
        value
    } else {
        return None;
    };

    let rounded = hz.round() as u32;
    if !(8_000..=384_000).contains(&rounded) {
        return None;
    }

    Some(rounded)
}

fn sample_rate_variants(path: &Path) -> Vec<(u32, PathBuf)> {
    let mut variants = Vec::new();

    let entries = match fs::read_dir(path) {
        Ok(entries) => entries,
        Err(_) => return variants,
    };

    for entry in entries.flatten() {
        let candidate = entry.path();
        if !candidate.is_dir() {
            continue;
        }

        let Some(name) = candidate.file_name().and_then(|n| n.to_str()) else {
            continue;
        };

        let Some(sample_rate) = parse_sample_rate_folder_name(name) else {
            continue;
        };

        if contains_wav_files(&candidate) {
            variants.push((sample_rate, candidate));
        }
    }

    variants.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| sort_key(&a.1).cmp(&sort_key(&b.1))));
    variants
}

pub fn canonicalize_sample_rate_root(path: &Path) -> PathBuf {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return path.to_path_buf();
    };

    if parse_sample_rate_folder_name(name).is_none() {
        return path.to_path_buf();
    }

    let Some(parent) = path.parent() else {
        return path.to_path_buf();
    };

    if sample_rate_variants(parent).is_empty() {
        return path.to_path_buf();
    }

    parent.to_path_buf()
}

pub fn resolve_sample_rate_variant_path(path: &Path, target_sr: u32) -> PathBuf {
    let canonical = canonicalize_sample_rate_root(path);

    if contains_wav_files(&canonical) {
        return canonical;
    }

    let variants = sample_rate_variants(&canonical);
    if variants.is_empty() {
        return canonical;
    }

    let chosen = variants.into_iter().min_by(|(a_sr, _), (b_sr, _)| {
        let a_diff = a_sr.abs_diff(target_sr);
        let b_diff = b_sr.abs_diff(target_sr);

        match a_diff.cmp(&b_diff) {
            Ordering::Equal => b_sr.cmp(a_sr),
            other => other,
        }
    });

    chosen.map(|(_, path)| path).unwrap_or(canonical)
}

fn build_node(path: &Path, rel_path: &str) -> Result<Option<PresetTreeNode>> {
    let mut has_ir_group = false;
    let mut child_dirs = Vec::new();

    for entry in fs::read_dir(path).with_context(|| format!("read_dir failed: {:?}", path))? {
        let entry = entry?;
        let entry_path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            if entry
                .file_name()
                .to_string_lossy()
                .starts_with('.')
            {
                continue;
            }
            child_dirs.push(entry_path);
        } else if file_type.is_file() && is_wav_file(&entry_path) {
            has_ir_group = true;
        }
    }

    let sample_rate_variants = sample_rate_variants(path);
    if !sample_rate_variants.is_empty() {
        has_ir_group = true;
    }

    let sample_rate_variant_paths: HashSet<PathBuf> = sample_rate_variants
        .into_iter()
        .map(|(_, variant_path)| variant_path)
        .collect();

    child_dirs.sort_by_key(|p| sort_key(p));

    let mut children = Vec::new();
    for child_dir in child_dirs {
        if sample_rate_variant_paths.contains(&child_dir) {
            continue;
        }

        let child_name = child_dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        if child_name.is_empty() {
            continue;
        }
        let child_rel = if rel_path.is_empty() {
            child_name.clone()
        } else {
            format!("{}/{}", rel_path, child_name)
        };
        if let Some(child_node) = build_node(&child_dir, &child_rel)? {
            children.push(child_node);
        }
    }

    if !has_ir_group && children.is_empty() {
        return Ok(None);
    }

    let name = if rel_path.is_empty() {
        "ROOT".to_string()
    } else {
        path.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| rel_path.to_string())
    };

    Ok(Some(PresetTreeNode {
        name,
        rel_path: normalize_leaf_id(rel_path),
        has_ir_group,
        children,
    }))
}

pub fn build_preset_tree(root: &Path) -> Result<PresetTreeNode> {
    if !root.exists() {
        return Err(anyhow::anyhow!("Preset root does not exist: {:?}", root));
    }
    if !root.is_dir() {
        return Err(anyhow::anyhow!(
            "Preset root is not a directory: {:?}",
            root
        ));
    }

    if let Some(node) = build_node(root, "")? {
        Ok(node)
    } else {
        Ok(PresetTreeNode::new_root())
    }
}

pub fn preset_pack_path(root: &Path) -> PathBuf {
    root.join(PRESET_PACK_FILE_NAME)
}

pub fn load_preset_pack(root: &Path) -> Result<PresetPack> {
    let path = preset_pack_path(root);
    if !path.exists() {
        return Ok(PresetPack::default());
    }

    let content = fs::read_to_string(&path)
        .with_context(|| format!("Failed to read preset pack: {:?}", path))?;
    let mut pack: PresetPack = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse preset pack JSON: {:?}", path))?;

    if pack.version == 0 {
        pack.version = default_pack_version();
    }

    let mut normalized_groups = BTreeMap::new();
    for (leaf_id, group) in pack.groups {
        normalized_groups.insert(normalize_leaf_id(&leaf_id), group);
    }
    pack.groups = normalized_groups;

    Ok(pack)
}

pub fn resolve_leaf_path(root: &Path, leaf_id: &str) -> PathBuf {
    let normalized = normalize_leaf_id(leaf_id);
    if normalized.is_empty() {
        return root.to_path_buf();
    }

    normalized
        .split('/')
        .fold(root.to_path_buf(), |acc, part| acc.join(part))
}

pub fn leaf_label(leaf_id: &str) -> String {
    normalize_leaf_id(leaf_id)
        .split('/')
        .last()
        .map(str::to_string)
        .unwrap_or_else(|| "Preset".to_string())
}
