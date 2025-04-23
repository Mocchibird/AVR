import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

from model import AVRModel_complex
from renderer import AVRRender

def generate_custom_rir(config_path, checkpoint_path, rx_positions, tx_positions, tx_directions=None, output_dir="generated_rirs"):
    """Generate RIRs for custom positions using a trained model
    
    Parameters:
    -----------
    config_path: Path to the config file
    checkpoint_path: Path to the model checkpoint
    rx_positions: List of receiver positions [[x1,y1,z1], [x2,y2,z2], ...]
    tx_positions: List of transmitter positions
    tx_directions: List of transmitter directions (optional)
    output_dir: Directory to save generated RIRs
    """
    # Load config
    with open(config_path, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    
    # Create model
    audionerf = AVRModel_complex(cfg['model'])
    renderer = AVRRender(networks_fn=audionerf, **cfg['render'])
    renderer = renderer.cuda()
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cuda')
    renderer.load_state_dict(ckpt['audionerf_network_state_dict'])
    renderer.eval()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert positions to tensors
    rx_positions = torch.tensor(rx_positions, dtype=torch.float32).cuda()
    tx_positions = torch.tensor(tx_positions, dtype=torch.float32).cuda()
    
    if tx_directions is not None:
        tx_directions = torch.tensor(tx_directions, dtype=torch.float32).cuda()
    
    # Generate RIRs
    with torch.no_grad():
        if tx_directions is not None:
            pred_sig = renderer(rx_positions, tx_positions, tx_directions)
        else:
            pred_sig = renderer(rx_positions, tx_positions)
        
        # Convert complex frequency domain to time domain
        pred_sig = pred_sig[..., 0] + 1j * pred_sig[..., 1]
        pred_time = torch.real(torch.fft.irfft(pred_sig, dim=-1))
    
    # Save results
    rirs = pred_time.cpu().numpy()
    fs = cfg['render']['fs']
    
    for i in range(len(rx_positions)):
        # Save as WAV file
        normalized_rir = rirs[i] / np.max(np.abs(rirs[i]))
        wavfile.write(os.path.join(output_dir, f"rir_rx{i}_tx{i}.wav"), fs, normalized_rir)
        
        # Plot and save figure
        plt.figure(figsize=(12, 6))
        plt.plot(normalized_rir)
        plt.title(f'RIR: Rx at {rx_positions[i].cpu().numpy()}, Tx at {tx_positions[i].cpu().numpy()}')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"rir_rx{i}_tx{i}.png"))
        plt.close()
    
    return rirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate custom RIRs using trained AVR model")
    parser.add_argument("--config", type=str, default="config_files/avr_raf_furnished.yml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, default="logs/RAF_Furnished/raf_test/ckpts/080000.tar", help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="generated_rirs", help="Directory to save output")
    args = parser.parse_args()
    
    # Example positions - replace these with your desired locations
    rx_positions = [
        [0.0, 0.0, 1.5],    # Example receiver position 1
        [1.0, 1.0, 1.5],    # Example receiver position 2
    ]
    
    tx_positions = [
        [2.0, 2.0, 1.7],    # Example transmitter position 1
        [3.0, 1.0, 1.7],    # Example transmitter position 2
    ]
    
    tx_directions = [
        [-1.0, -1.0, 0.0],  # Example direction 1 (normalized)
        [-1.0, 0.0, 0.0],   # Example direction 2 (normalized)
    ]
    
    generate_custom_rir(
        args.config,
        args.checkpoint,
        rx_positions,
        tx_positions,
        tx_directions,
        args.output_dir
    )
