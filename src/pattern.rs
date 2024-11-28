// src/pattern.rs
use crate::utils::*;
use image::{ImageBuffer, Rgb};
use ndarray::{Array1, Array2, Axis};

/// Create a meshgrid from x and y arrays, similar to numpy's meshgrid
pub fn meshgrid(x: &Array1<f64>, y: &Array1<f64>) -> (Array2<f64>, Array2<f64>) {
    let nx = x.len();
    let ny = y.len();

    // Create x grid
    let mut x_grid = Array2::zeros((ny, nx));
    for i in 0..ny {
        x_grid.row_mut(i).assign(x);
    }

    // Create y grid
    let mut y_grid = Array2::zeros((ny, nx));
    for j in 0..nx {
        y_grid.column_mut(j).assign(y);
    }

    (x_grid, y_grid)
}

pub fn generate_strawberry_pattern(width: u32, height: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    // Create coordinate arrays
    let x: Array1<f64> = Array1::linspace(-999.0 / 600.0, 1000.0 / 600.0, width as usize);
    let mut y: Array1<f64> = Array1::linspace(-599.0 / 600.0, 600.0 / 600.0, height as usize);
    y.invert_axis(Axis(0));
    let (x_grid, y_grid) = meshgrid(&x, &y);

    // Initialize cache
    // let cache = HVCache::new(&x_grid, &y_grid, 31);

    // Generate all color channels in parallel
    let channels: Vec<Array2<u8>> = vec![0.0, 1.0, 2.0]
        .into_iter()
        .map(|v| {
            let h = h_v(&x_grid, &y_grid, v);
            f_vectorized(&h)
        })
        .collect();

    // Create and fill image buffer
    let mut img = ImageBuffer::new(width, height);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        *pixel = Rgb([
            channels[0][[y as usize, x as usize]],
            channels[1][[y as usize, x as usize]],
            channels[2][[y as usize, x as usize]],
        ]);
    }
    img
}

pub fn generate_strawberry_pattern_old(width: u32, height: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    // Create coordinate arrays
    let x: Array1<f64> = Array1::linspace(-999.0 / 600.0, 1000.0 / 600.0, width as usize);
    let mut y: Array1<f64> = Array1::linspace(-599.0 / 600.0, 600.0 / 600.0, height as usize);
    y.invert_axis(Axis(0));

    let (x_grid, y_grid) = meshgrid(&x, &y);

    println!("Generating color channels...");

    // Generate red channel (v=0)
    println!("Generating red channel...");
    let h0 = h_v(&x_grid, &y_grid, 0.0);
    let red: Array2<u8> = h0.mapv(f);

    // Generate green channel (v=1)
    println!("Generating green channel...");
    let h1 = h_v(&x_grid, &y_grid, 1.0);
    let green: Array2<u8> = h1.mapv(f);

    // Generate blue channel (v=2)
    println!("Generating blue channel...");
    let h2 = h_v(&x_grid, &y_grid, 2.0);
    let blue: Array2<u8> = h2.mapv(f);

    // Create image buffer
    let mut img = ImageBuffer::new(width, height);

    // Fill image buffer with RGB values
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        *pixel = Rgb([
            red[[y as usize, x as usize]],
            green[[y as usize, x as usize]],
            blue[[y as usize, x as usize]],
        ]);
    }

    img
}
