use std::fs;
use serde::Deserialize;


#[derive(Deserialize,Debug)]
pub struct Support{

     layers: u32,
     neurons: [Ns;5],
     edges: [Es;12],

}
#[derive(Debug,Deserialize)]
struct Ns{
    id : u32,
    layer: u32,
}
#[derive(Debug,Deserialize)]
struct Es{
    source_id: u32,
    dest_id: u32,
    weight: f64,
}


impl Support {

    pub fn from_json(path: &str)->Support{

        let read= fs::read_to_string(path).expect("something went wrong with the file reading");
        println!("ho letto {read}");

        let deserialized:Support= serde_json::from_str(&read.as_str()).expect("something went wrong deserializing the struct");
        return deserialized;

            
    }
    
}