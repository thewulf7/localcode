use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "skills/"]
pub struct SkillsAssets;

fn main() {
    for file in SkillsAssets::iter() {
        let path = file.as_ref();
        println!("Embedded Path: {}", path);
    }
}
