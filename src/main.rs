mod pattern;
mod utils;

use pattern::generate_strawberry_pattern;

fn main() {
    println!("Generating strawberry pattern...");
    let img = generate_strawberry_pattern(500, 300);

    println!("Saving image...");
    img.save("generated_strawberries.png").unwrap();
    println!("Image saved as generated_strawberries.png");
}
