// main.rs

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use serde::{Deserialize, Serialize};
use log::{info, error, warn, debug};
use env_logger;
use ocl::{self, Platform, Device, Context, Queue, Program, Kernel, Buffer, flags};
use serde_json;
use std::path::Path;

// Removed unused imports
// use bip39::{Mnemonic, Language};

#[derive(Debug, Clone)]
struct WorkResponse {
    indices: Vec<u32>,
    offset: u64,
    batch_size: usize,
}

#[derive(Debug, Clone)]
struct Work {
    start_hi: u32,
    start_lo: u32,
    batch_size: usize,
    offset: u64,
}

// Declare the PROGRESS_FILE constant at the module level
const PROGRESS_FILE: &str = "progress.json";

fn sweep_btc(mnemonic: &str) {
    info!("Sweeping BTC for mnemonic: {}", mnemonic);
    // TODO: Implement sweep_btc functionality
}

fn broadcast_tx(rawtx: &str) {
    info!("Broadcasting transaction: {}", rawtx);
    // TODO: Implement broadcast_tx functionality
}

fn log_solution(offset: u64, mnemonic: &str) {
    info!("Solution found at offset {}: {}", offset, mnemonic);
    // Optionally, write to a separate file
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("solutions.log")
        .unwrap();
    writeln!(file, "{}: {}", offset, mnemonic).unwrap();
}

fn log_work(offset: u64) {
    debug!("Work logged at offset {}", offset);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("work.log")
        .unwrap();
    writeln!(file, "{}", offset).unwrap();
}

#[derive(Serialize, Deserialize)]
struct ProgressData {
    current_offset: u64,
}

fn initialize_progress() -> u64 {
    if let Ok(mut file) = File::open(PROGRESS_FILE) {
        let mut data = String::new();
        if let Err(e) = file.read_to_string(&mut data) {
            error!("Error reading progress file {}: {}", PROGRESS_FILE, e);
            0
        } else {
            match serde_json::from_str::<ProgressData>(&data) {
                Ok(progress) => {
                    info!("Resuming work from offset {}", progress.current_offset);
                    progress.current_offset
                }
                Err(e) => {
                    error!("Error parsing progress file {}: {}", PROGRESS_FILE, e);
                    0
                }
            }
        }
    } else {
        0
    }
}

fn save_progress(current_offset: u64) {
    let progress = ProgressData { current_offset };
    let data = serde_json::to_string(&progress).unwrap();
    if let Err(e) = fs::write(PROGRESS_FILE, data) {
        error!("Error saving progress to {}: {}", PROGRESS_FILE, e);
    } else {
        debug!("Progress saved at offset {}", current_offset);
    }
}

struct WorkState {
    current_offset: u64,
}

fn get_work(
    work_state: Arc<Mutex<WorkState>>,
    total_addresses: u64,
    batch_size: usize,
) -> Option<Work> {
    let mut state = work_state.lock().unwrap();
    if state.current_offset >= total_addresses {
        None
    } else {
        let work_offset = state.current_offset;
        state.current_offset += batch_size as u64;

        // Save progress after allocating work
        save_progress(state.current_offset);

        // Calculate start_hi and start_lo based on work_offset
        // Assuming 32 bits for hi and lo parts
        let start_hi = ((work_offset >> 32) & 0xFFFFFFFF) as u32;
        let start_lo = (work_offset & 0xFFFFFFFF) as u32;

        Some(Work {
            start_hi,
            start_lo,
            batch_size,
            offset: work_offset,
        })
    }
}

fn mnemonic_gpu(
    platform_name: &str,
    device_name: &str,
    kernel_source: &str,
    kernel_name: &str,
    work_state: Arc<Mutex<WorkState>>,
    total_addresses: u64,
    batch_size: usize,
) {
    // Get all platforms
    let platforms = Platform::list();

    info!("[{}] Available Platforms:", device_name);
    for p in &platforms {
        info!("    Platform: {:?}", p.name().unwrap());
    }

    // Find the specified platform
    let platform = platforms.into_iter().find(|p| p.name().unwrap() == platform_name);
    if platform.is_none() {
        error!("[{}] Platform '{}' not found.", device_name, platform_name);
        return;
    }
    let platform = platform.unwrap();

    // Find the specified device within the platform
    let devices = Device::list(platform, None).unwrap();
    let device = devices.into_iter().find(|d| d.name().unwrap() == device_name);
    if device.is_none() {
        error!(
            "[{}] Device '{}' not found on platform '{}'.",
            device_name, device_name, platform_name
        );
        return;
    }
    let device = device.unwrap();

    info!("[{}] Initializing OpenCL context.", device_name);
    let context = Context::builder()
        .platform(platform)
        .devices(device.clone())
        .build()
        .unwrap();

    let queue = Queue::new(&context, device.clone(), None).unwrap();

    info!("[{}] Building OpenCL program.", device_name);
    let program_build = Program::builder()
        .src(kernel_source)
        .devices(device.clone())
        .build(&context);

    let program = match program_build {
        Ok(p) => p,
        Err(e) => {
            error!("[{}] OpenCL Program Build Failed: {}", device_name, e);
            return;
        }
    };

    info!(
        "[{}] Program built successfully. Starting processing loop.",
        device_name
    );

    const MAX_MNEMONIC_LENGTH: usize = 256; // Adjust as needed

    // Processing loop
    loop {
        let work = {
            let state = work_state.clone();
            get_work(state, total_addresses, batch_size)
        };

        if work.is_none() {
            info!("[{}] No more work available. Exiting...", device_name);
            break;
        }
        let work = work.unwrap();

        let items = work.batch_size;
        let mnemonic_hi = work.start_hi;
        let mnemonic_lo = work.start_lo;

        // Prepare buffers
        let mut target_mnemonic = vec![0u8; MAX_MNEMONIC_LENGTH];
        let mut mnemonic_found = vec![0u8; 1];
        let mut addresses_checked = vec![0u8; items * 34];
        let mut mnemonics_buffer = vec![0u8; items * MAX_MNEMONIC_LENGTH];

        // Create buffers
        let target_mnemonic_buf = Buffer::<u8>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_WRITE)
            .len(target_mnemonic.len())
            .copy_host_slice(&target_mnemonic)
            .build()
            .unwrap();

        let mnemonic_found_buf = Buffer::<u8>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_WRITE)
            .len(mnemonic_found.len())
            .copy_host_slice(&mnemonic_found)
            .build()
            .unwrap();

        let addresses_checked_buf = Buffer::<u8>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_WRITE_ONLY)
            .len(addresses_checked.len())
            .build()
            .unwrap();

        let mnemonics_buf = Buffer::<u8>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_WRITE_ONLY)
            .len(mnemonics_buffer.len())
            .build()
            .unwrap();

        // Calculate the seed
        let combined_seed = ((mnemonic_hi as u64) << 32) | (mnemonic_lo as u64);
        debug!(
            "[{}] Combined Seed: {} (0x{:016X})",
            device_name, combined_seed, combined_seed
        );

        let seed = combined_seed as u32;

        // Create kernel and specify arguments during build
        let kernel = match Kernel::builder()
            .program(&program)
            .name(kernel_name)
            .queue(queue.clone())
            .global_work_size(items)
            .arg(seed)
            .arg(&target_mnemonic_buf)
            .arg(&mnemonic_found_buf)
            .arg(&addresses_checked_buf)
            .arg(&mnemonics_buf)
            .build()
        {
            Ok(k) => k,
            Err(e) => {
                error!(
                    "[{}] Error creating kernel '{}': {}",
                    device_name, kernel_name, e
                );
                continue;
            }
        };

        // Execute kernel
        let enqueue_result = unsafe { kernel.enq() };

        if let Err(e) = enqueue_result {
            error!("[{}] Error executing kernel: {}", device_name, e);
            continue;
        }

        // Read buffers
        let read_result = target_mnemonic_buf
            .read(&mut target_mnemonic)
            .enq()
            .and_then(|_| mnemonic_found_buf.read(&mut mnemonic_found).enq())
            .and_then(|_| addresses_checked_buf.read(&mut addresses_checked).enq())
            .and_then(|_| mnemonics_buf.read(&mut mnemonics_buffer).enq());

        if let Err(e) = read_result {
            error!("[{}] Error reading buffers: {}", device_name, e);
            continue;
        }

        // Log the work offset
        log_work(work.offset);

        // Write checked addresses to a file
        let safe_device_name = device_name.replace(' ', "_").replace('/', "_");
        let addresses_file_name = format!("addresses_checked_{}.txt", safe_device_name);

        if let Ok(mut addr_file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&addresses_file_name)
        {
            for i in 0..items {
                let address_start = i * 34;
                let address_end = address_start + 34;
                let address_bytes = &addresses_checked[address_start..address_end];
                let address_str = String::from_utf8_lossy(address_bytes)
                    .trim_end_matches('\0')
                    .to_string();
                if let Err(e) = writeln!(addr_file, "{}", address_str) {
                    error!(
                        "[{}] Error writing address {}: {}",
                        device_name, i, e
                    );
                }
            }
        } else {
            error!(
                "[{}] Error opening addresses file: {}",
                device_name, addresses_file_name
            );
        }

        // Write mnemonics to a file
        let mnemonics_file_name = format!("mnemonics_{}.txt", safe_device_name);

        if let Ok(mut mnemonic_file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&mnemonics_file_name)
        {
            for i in 0..items {
                let mnemonic_start = i * MAX_MNEMONIC_LENGTH;
                let mnemonic_end = mnemonic_start + MAX_MNEMONIC_LENGTH;
                let mnemonic_bytes = &mnemonics_buffer[mnemonic_start..mnemonic_end];
                let mnemonic_str = String::from_utf8_lossy(mnemonic_bytes)
                    .trim_end_matches('\0')
                    .to_string();
                if !mnemonic_str.is_empty() {
                    if let Err(e) = writeln!(mnemonic_file, "{}", mnemonic_str) {
                        error!(
                            "[{}] Error writing mnemonic {}: {}",
                            device_name, i, e
                        );
                    }
                }
            }
        } else {
            error!(
                "[{}] Error opening mnemonics file: {}",
                device_name, mnemonics_file_name
            );
        }

        // Check if any mnemonic was found
        if mnemonic_found[0] == 0x01 {
            let mnemonic = String::from_utf8_lossy(&target_mnemonic)
                .trim_end_matches('\0')
                .to_string();
            info!(
                "[{}] Mnemonic found at offset {}: {}",
                device_name, work.offset, mnemonic
            );
            log_solution(work.offset, &mnemonic);
            sweep_btc(&mnemonic);
        }
    }
}

fn load_kernel_files(file_list: &[&str]) -> String {
    let mut kernel_source = String::new();
    for file in file_list {
        let file_path = format!("cl/{}.cl", file);
        if !Path::new(&file_path).exists() {
            error!("Kernel file {} does not exist.", file_path);
            continue;
        }
        match fs::read_to_string(&file_path) {
            Ok(content) => {
                kernel_source.push_str(&content);
                kernel_source.push('\n');
                debug!("Loaded kernel file: {}", file_path);
            }
            Err(e) => {
                error!("Error reading kernel file {}: {}", file_path, e);
            }
        }
    }
    kernel_source
}

fn test_opencl() {
    let platforms = Platform::list();

    info!("Available OpenCL Platforms:");
    for platform in &platforms {
        info!("  Platform: {:?}", platform.name().unwrap());
        match Device::list(platform.clone(), Some(flags::DEVICE_TYPE_GPU)) {
            Ok(devices) => {
                for device in devices {
                    info!("    Device: {:?}", device.name().unwrap());
                }
            }
            Err(e) => {
                error!(
                    "Error fetching devices for platform '{:?}': {}",
                    platform.name().unwrap(),
                    e
                );
            }
        }
    }

    if platforms.is_empty() {
        error!("No OpenCL platforms found.");
        return;
    }

    // Select the first GPU device
    let platform = platforms[0];
    match Device::list(platform.clone(), Some(flags::DEVICE_TYPE_GPU)) {
        Ok(devices) => {
            if devices.is_empty() {
                error!("No GPU devices found on the first platform.");
                return;
            }
            let device = devices[0].clone();
            info!("Selected Platform: {:?}", platform.name().unwrap());
            info!("Selected Device: {:?}", device.name().unwrap());

            // Create context and queue
            let context = Context::builder()
                .platform(platform.clone())
                .devices(device.clone())
                .build()
                .unwrap();

            let queue = Queue::new(&context, device.clone(), None).unwrap();

            // Simple kernel to add two numbers
            let program_source = r#"
            __kernel void add(__global const float *a, __global const float *b, __global float *c) {
                int gid = get_global_id(0);
                c[gid] = a[gid] + b[gid];
            }
            "#;
            let program = Program::builder()
                .src(program_source)
                .devices(device.clone())
                .build(&context)
                .unwrap();

            let a = vec![1.0f32, 2.0, 3.0, 4.0];
            let b = vec![5.0f32, 6.0, 7.0, 8.0];
            let mut c = vec![0.0f32; a.len()];

            let a_buf = Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
                .len(a.len())
                .copy_host_slice(&a)
                .build()
                .unwrap();

            let b_buf = Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
                .len(b.len())
                .copy_host_slice(&b)
                .build()
                .unwrap();

            let c_buf = Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(flags::MEM_WRITE_ONLY)
                .len(c.len())
                .build()
                .unwrap();

            let kernel = Kernel::builder()
                .program(&program)
                .name("add")
                .queue(queue.clone())
                .global_work_size(a.len())
                .arg(&a_buf)
                .arg(&b_buf)
                .arg(&c_buf)
                .build()
                .unwrap();

            unsafe {
                kernel.enq().unwrap();
            }

            c_buf.read(&mut c).enq().unwrap();

            info!("Simple OpenCL Test Result: {:?}", c);
        }
        Err(e) => {
            error!(
                "Error fetching devices for platform '{:?}': {}",
                platform.name().unwrap(),
                e
            );
        }
    }
}

fn main() {
    env_logger::init();

    info!("Running OpenCL Test...");
    test_opencl();
    info!("OpenCL Test Completed.\n");

    // Load OpenCL kernel files
    let int_to_address_files = vec![
        "common",
        "ripemd",
        "sha2",
        "secp256k1_common",
        "secp256k1_scalar",
        "secp256k1_field",
        "secp256k1_group",
        "secp256k1_prec",
        "secp256k1",
        "address",
        "mnemonic_constants",
        "int_to_address",
    ];
    let kernel_name = "int_to_address";
    let kernel_source = load_kernel_files(&int_to_address_files);

    if kernel_source.trim().is_empty() {
        error!("No kernel source loaded. Please check the kernel files. Exiting.");
        return;
    }

    info!("Kernel source loaded successfully.");

    // Initialize shared state
    let work_state = Arc::new(Mutex::new(WorkState {
        current_offset: initialize_progress(),
    }));

    // Define total addresses and batch size
    const TOTAL_ADDRESSES: u64 = 1 << 32; // 2^32
    const BATCH_SIZE: usize = 1024;

    // Get OpenCL platforms and devices
    let platforms = Platform::list();
    if platforms.is_empty() {
        error!("No OpenCL platforms found. Exiting.");
        return;
    }

    info!("\nAvailable OpenCL Platforms and GPU Devices:");
    let mut devices_info = Vec::new();
    for platform in &platforms {
        info!("Platform: {:?}", platform.name().unwrap());
        match Device::list(platform.clone(), Some(flags::DEVICE_TYPE_GPU)) {
            Ok(devices) => {
                if devices.is_empty() {
                    warn!(
                        "  No GPU devices found on platform '{:?}'.",
                        platform.name().unwrap()
                    );
                    continue;
                }
                for device in devices {
                    info!("  Device: {:?}", device.name().unwrap());
                    devices_info.push((platform.clone(), device));
                }
            }
            Err(e) => {
                error!(
                    "  Error fetching devices for platform '{:?}': {}",
                    platform.name().unwrap(),
                    e
                );
            }
        }
    }

    if devices_info.is_empty() {
        error!("No GPU devices found across all platforms. Exiting.");
        return;
    }

    // Start a thread for each GPU device
    let mut handles = Vec::new();
    for (platform, device) in devices_info {
        let platform_name = platform.name().unwrap();
        let device_name = device.name().unwrap();
        info!(
            "\nStarting thread for device: '{}' on platform: '{}'",
            device_name, platform_name
        );
        let kernel_source_clone = kernel_source.clone();
        let work_state_clone = Arc::clone(&work_state);

        let kernel_name_clone = kernel_name.to_string();

        let handle = thread::Builder::new()
            .name(device_name.clone())
            .spawn(move || {
                mnemonic_gpu(
                    &platform_name,
                    &device_name,
                    &kernel_source_clone,
                    &kernel_name_clone,
                    work_state_clone,
                    TOTAL_ADDRESSES,
                    BATCH_SIZE,
                );
            })
            .unwrap();
        handles.push(handle);
    }

    // Optionally, join the threads
    for handle in handles {
        handle.join().unwrap();
    }
}
