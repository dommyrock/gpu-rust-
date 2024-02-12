use std::process::Command;

fn main() {
   //nvcc cuda.cu -o cuda.exe
    let output = Command::new("nvcc")
        .arg("src/cuda.cu")
        .arg("-o")
        .arg("cuda.exe")
        .output()
        .expect("Failed to execute command");

    if !output.status.success() {
        let error_message = String::from_utf8_lossy(&output.stderr);
        panic!("Build failed with error:\n{}", error_message);
    }
}

//bindgen_cuda
// fn main() {
//     let builder = bindgen_cuda::Builder::default();
//     let bindings = builder.build_ptx().unwrap();
//     bindings.write("src/lib.rs");
// }