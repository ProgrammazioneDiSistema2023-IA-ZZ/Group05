use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;

#[derive(Serialize, Deserialize)]
pub struct InputMatrix(pub Vec<Vec<bool>>);

fn generate_input(rows: usize, cols: usize, probability: f64) -> InputMatrix {
    let mut rng = rand::thread_rng();

    let mut matrix: Vec<Vec<bool>> = vec![];

    for _ in 0..rows {
        let row: Vec<bool> = (0..cols).map(|_| rng.gen::<f64>() < probability).collect();
        matrix.push(row);
    }

    InputMatrix(matrix)
}

fn main() {
    // Crea la matrice con i dati
    let matrix = generate_input(6, 10, 0.85);

    // Serializza la matrice in JSON
    let serialized_matrix =
        serde_json::to_string(&matrix).expect("Errore durante la serializzazione.");

    // Scrivi i dati serializzati su un file JSON
    let mut file = File::create("sources\\simulation_input.json")
        .expect("Errore durante l'apertura del file.");
    file.write_all(serialized_matrix.as_bytes())
        .expect("Errore durante la scrittura del file.");
}
