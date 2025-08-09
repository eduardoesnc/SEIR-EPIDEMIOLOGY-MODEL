import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.integrate import odeint
from numba import jit
from PIL import Image
import glob

def seir_equations(y, t, N, beta, gamma, sigma):
    S, E, I, R = y
    dS_dt = -beta * S * I / N
    dE_dt = beta * S * I / N - sigma * E
    dI_dt = sigma * E - gamma * I
    dR_dt = gamma * I
    
    return dS_dt, dE_dt, dI_dt, dR_dt

def run_seir_model(N, E0, I0, R0, beta, gamma, sigma, total_time, time_steps):
    S0 = N - E0 - I0 - R0
    y0 = S0, E0, I0, R0
    t = np.linspace(0, total_time, time_steps)
    
    solution = odeint(seir_equations, y0, t, args=(N, beta, gamma, sigma))
    S, E, I, R = solution.T
    
    return t, S, E, I, R

SUSCEPTIBLE = 0
EXPOSED = 1
INFECTED = 2
RECOVERED = 3

@jit(nopython=True)
def run_ca_simulation_seir(L, time_steps, sigma_ca, gamma_ca):
    grid = np.full(L, SUSCEPTIBLE, dtype=np.int8)
    
    centers = [L//3, 2*L//3]
    for center in centers:
        grid[center] = INFECTED
        for offset in [-2, -1, 1, 2]:
            pos = (center + offset) % L
            grid[pos] = EXPOSED
        for offset in [-3, 3]:
            pos = (center + offset) % L
            if np.random.rand() < 0.7:
                grid[pos] = INFECTED
    
    grid_history = np.zeros((time_steps, L), dtype=np.int8)
    grid_history[0, :] = grid
    
    for t in range(1, time_steps):
        new_grid = grid.copy()
        
        for i in range(L):
            state = grid[i]
            
            if state == SUSCEPTIBLE:
                infected_neighbors = 0
                exposed_neighbors = 0
                
                for offset in [-2, -1, 1, 2]:
                    neighbor_pos = (i + offset) % L
                    neighbor_state = grid[neighbor_pos]
                    if neighbor_state == INFECTED:
                        infected_neighbors += 1
                    elif neighbor_state == EXPOSED:
                        exposed_neighbors += 1
                
                if infected_neighbors > 0:
                    prob = min(0.9 * infected_neighbors, 0.98)
                    if np.random.rand() < prob:
                        new_grid[i] = EXPOSED
                elif exposed_neighbors > 0:
                    prob = min(0.6 * exposed_neighbors, 0.85)
                    if np.random.rand() < prob:
                        new_grid[i] = EXPOSED
                elif np.random.rand() < 0.002:
                    new_grid[i] = EXPOSED
            
            elif state == EXPOSED:
                if np.random.rand() < sigma_ca:
                    new_grid[i] = INFECTED
            
            elif state == INFECTED:
                if np.random.rand() < gamma_ca:
                    new_grid[i] = RECOVERED
        
        grid = new_grid
        grid_history[t, :] = grid
    
    return grid_history

def calculate_spatial_entropy(grid_state, block_size):
    L = len(grid_state)
    num_blocks = L // block_size
    
    state_counts = np.zeros((num_blocks, 4), dtype=np.int32)
    for i in range(num_blocks):
        block = grid_state[i*block_size:(i+1)*block_size]
        for state in range(4):
            state_counts[i, state] = np.sum(block == state)

    probabilities = state_counts / block_size
    
    total_entropy = 0.0
    for i in range(num_blocks):
        block_entropy = 0.0
        for p in probabilities[i]:
            if p > 0:
                block_entropy -= p * np.log2(p)
        total_entropy += block_entropy
        
    return total_entropy / num_blocks

def calculate_global_entropy(grid_state):
    unique, counts = np.unique(grid_state, return_counts=True)
    probabilities = counts / len(grid_state)
    
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy

def plot_seir_curve(t, S, E, I, R, N, output_dir):
    plt.figure(figsize=(14, 8))
    plt.plot(t, S / N, label='Suscetíveis (S)', color='#4A90E2', linewidth=3)
    plt.plot(t, E / N, label='Expostos (E)', color='#FFA500', linewidth=3)
    plt.plot(t, I / N, label='Infectados (I)', color='#FF4444', linewidth=3)
    plt.plot(t, R / N, label='Recuperados (R)', color='#32CD32', linewidth=3)
    plt.xlabel('Tempo (dias)', fontsize=14)
    plt.ylabel('Proporção da População', fontsize=14)
    plt.title('Modelo SEIR - Dinâmica Epidêmica', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "seir_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_perfect_ca_visualization(history, output_dir):
    colors = ['#F0F8FF', '#FFD700', '#FF4500', '#228B22']
    cmap = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    

    plt.figure(figsize=(20, 12))
    plt.imshow(history, cmap=cmap, norm=norm, aspect='auto', interpolation='none')
    
    ax = plt.gca()
    for x in range(0, history.shape[1], 50):
        ax.axvline(x - 0.5, color='gray', linewidth=0.2, alpha=0.4)
    for y in range(0, history.shape[0], 20):
        ax.axhline(y - 0.5, color='gray', linewidth=0.2, alpha=0.4)
    
    plt.xlabel('Posição na Grade (células)', fontsize=16)
    plt.ylabel('Tempo (dias)', fontsize=16)
    plt.title('Autômato Celular SEIR - Evolução Espaço-Temporal Completa', fontsize=18, fontweight='bold')
    
    cbar = plt.colorbar(ticks=[0, 1, 2, 3], shrink=0.8)
    cbar.set_ticklabels(['Suscetível', 'Exposto', 'Infectado', 'Recuperado'])
    cbar.ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ca_perfect_evolution.png"), dpi=300, bbox_inches='tight')
    plt.close()

    center = history.shape[1] // 2
    zoom_range = 150
    zoom_start = max(0, center - zoom_range)
    zoom_end = min(history.shape[1], center + zoom_range)
    zoom_time = min(100, history.shape[0])
    
    zoomed = history[:zoom_time, zoom_start:zoom_end]
    
    plt.figure(figsize=(18, 10))
    plt.imshow(zoomed, cmap=cmap, norm=norm, aspect='auto', interpolation='none')
    
    ax = plt.gca()
    for x in range(0, zoomed.shape[1], 10):
        ax.axvline(x - 0.5, color='black', linewidth=0.3, alpha=0.6)
    for y in range(0, zoomed.shape[0], 5):
        ax.axhline(y - 0.5, color='black', linewidth=0.3, alpha=0.6)
    
    plt.xlabel('Posição na Grade (células)', fontsize=14)
    plt.ylabel('Tempo (dias)', fontsize=14)
    plt.title('Zoom Detalhado - Propagação das Ondas Epidêmicas', fontsize=16, fontweight='bold')
    
    cbar = plt.colorbar(ticks=[0, 1, 2, 3], shrink=0.8)
    cbar.set_ticklabels(['Suscetível', 'Exposto', 'Infectado', 'Recuperado'])
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ca_zoom_perfect.png"), dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(6, 1, figsize=(20, 15))
    snapshot_times = [5, 15, 30, 50, 80, 120]
    
    for i, t in enumerate(snapshot_times):
        if t < len(history):
            snapshot_2d = np.tile(history[t, :], (20, 1))
            im = axes[i].imshow(snapshot_2d, cmap=cmap, norm=norm, aspect='auto', interpolation='none')
            
            axes[i].set_title(f'Dia {t} - Padrão Espacial da Epidemia', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Posição na Grade (células)')
            axes[i].set_ylabel('Visualização')
            
            stats = np.bincount(history[t, :], minlength=4)
            total = len(history[t, :])
            stats_text = f'S: {stats[0]} ({stats[0]/total*100:.1f}%) | E: {stats[1]} ({stats[1]/total*100:.1f}%) | I: {stats[2]} ({stats[2]/total*100:.1f}%) | R: {stats[3]} ({stats[3]/total*100:.1f}%)'
            
            axes[i].text(0.02, 0.85, stats_text, transform=axes[i].transAxes, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                        fontsize=11, fontfamily='monospace')
    
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.6, pad=0.02)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['Suscetível', 'Exposto', 'Infectado', 'Recuperado'])
    
    plt.suptitle('Evolução Temporal - Snapshots da Propagação Epidêmica', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ca_snapshots_perfect.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_ca_statistics_perfect(history, t, output_dir):
    time_steps, L = history.shape
    susceptible_count = np.sum(history == SUSCEPTIBLE, axis=1)
    exposed_count = np.sum(history == EXPOSED, axis=1)
    infected_count = np.sum(history == INFECTED, axis=1)
    recovered_count = np.sum(history == RECOVERED, axis=1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    ax1.plot(t, susceptible_count/L, label='Suscetíveis', color='#4A90E2', linewidth=3)
    ax1.plot(t, exposed_count/L, label='Expostos', color='#FFA500', linewidth=3)
    ax1.plot(t, infected_count/L, label='Infectados', color='#FF4444', linewidth=3)
    ax1.plot(t, recovered_count/L, label='Recuperados', color='#32CD32', linewidth=3)
    ax1.set_xlabel('Tempo (dias)')
    ax1.set_ylabel('Proporção da População')
    ax1.set_title('Dinâmica Populacional no Autômato Celular', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, infected_count, color='#FF4444', linewidth=3)
    ax2.fill_between(t, infected_count, alpha=0.3, color='#FF4444')
    ax2.set_xlabel('Tempo (dias)')
    ax2.set_ylabel('Número de Infectados')
    ax2.set_title('Curva de Infectados no AC', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    front_width = []
    for time_step in range(time_steps):
        infected_pos = np.where(history[time_step, :] == INFECTED)[0]
        if len(infected_pos) > 0:
            width = np.max(infected_pos) - np.min(infected_pos)
            front_width.append(width)
        else:
            front_width.append(0)
    
    ax3.plot(t, front_width, color='purple', linewidth=3)
    ax3.set_xlabel('Tempo (dias)')
    ax3.set_ylabel('Largura da Frente (células)')
    ax3.set_title('Velocidade de Propagação Espacial', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    regions = 10
    region_size = L // regions
    density_matrix = np.zeros((time_steps, regions))
    
    for t_idx in range(time_steps):
        for region in range(regions):
            start = region * region_size
            end = min((region + 1) * region_size, L)
            region_infected = np.sum(history[t_idx, start:end] == INFECTED)
            density_matrix[t_idx, region] = region_infected / (end - start)
    
    im = ax4.imshow(density_matrix.T, cmap='Reds', aspect='auto', interpolation='nearest')
    ax4.set_xlabel('Tempo (dias)')
    ax4.set_ylabel('Região da Grade')
    ax4.set_title('Densidade de Infectados por Região', fontweight='bold')
    plt.colorbar(im, ax=ax4, label='Densidade')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ca_statistics_perfect.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_entropy_perfect(t, spatial_entropies, global_entropies, output_dir):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

    ax1.plot(t, spatial_entropies, color='#8A2BE2', linewidth=3, label='Entropia Espacial')
    ax1.fill_between(t, spatial_entropies, alpha=0.3, color='#8A2BE2')
    ax1.set_ylabel('Entropia Espacial (bits)')
    ax1.set_title('Evolução da Entropia Espacial', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(t, global_entropies, color='#FF6347', linewidth=3, label='Entropia Global')
    ax2.fill_between(t, global_entropies, alpha=0.3, color='#FF6347')
    ax2.set_ylabel('Entropia Global (bits)')
    ax2.set_title('Evolução da Entropia Global', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.plot(t, spatial_entropies, color='#8A2BE2', linewidth=3, label='Entropia Espacial')
    ax3.plot(t, global_entropies, color='#FF6347', linewidth=3, label='Entropia Global')
    ax3.set_xlabel('Tempo (dias)')
    ax3.set_ylabel('Entropia (bits)')
    ax3.set_title('Comparação das Entropias', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "entropy_perfect.png"), dpi=300, bbox_inches='tight')
    plt.close()


def create_dramatic_gif_frames(history, t, spatial_entropies, global_entropies, output_dir):
    colors = ['#000033', '#FFD700', '#FF1493', '#00FF00']
    cmap = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    gif_frames_dir = os.path.join(output_dir, "gif_frames")
    if not os.path.exists(gif_frames_dir):
        os.makedirs(gif_frames_dir)
    
    frame_interval = 2
    frame_count = 0
    
    for day in range(0, len(history), frame_interval):
        if day >= len(history):
            break
            
        fig = plt.figure(figsize=(16, 12), facecolor='black')
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], width_ratios=[3, 1])
        
        ax_main = fig.add_subplot(gs[0, :])
        current_state = np.tile(history[day, :], (60, 1))
        im_main = ax_main.imshow(current_state, cmap=cmap, norm=norm, aspect='auto', interpolation='none')
        
        ax_main.set_title(f'🦠 EPIDEMIA SEIR - DIA {day} 🦠', 
                         fontsize=24, fontweight='bold', color='white', pad=20)
        ax_main.set_xlabel('Posição na Grade (células)', fontsize=14, color='white')
        ax_main.set_ylabel('Visualização Ampliada', fontsize=14, color='white')
        
        ax_main.set_xticks([])
        ax_main.set_yticks([])
        for spine in ax_main.spines.values():
            spine.set_color('white')
            spine.set_linewidth(2)

        ax_stats = fig.add_subplot(gs[1, 0])
        ax_stats.set_facecolor('black')

        day_stats = np.bincount(history[day, :], minlength=4)
        total_cells = len(history[day, :])

        states = ['Suscetível', 'Exposto', 'Infectado', 'Recuperado']
        percentages = day_stats / total_cells * 100
        bars = ax_stats.bar(states, percentages, color=colors, alpha=0.8, edgecolor='white', linewidth=2)

        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax_stats.text(bar.get_x() + bar.get_width()/2., height + 1,
                         f'{pct:.1f}%', ha='center', va='bottom', 
                         fontweight='bold', color='white', fontsize=12)
        
        ax_stats.set_title('📊 DISTRIBUIÇÃO POPULACIONAL', fontweight='bold', color='white', fontsize=14)
        ax_stats.set_ylabel('Porcentagem (%)', color='white')
        ax_stats.set_ylim(0, 100)
        ax_stats.tick_params(colors='white')
        
        for spine in ax_stats.spines.values():
            spine.set_color('white')

        ax_entropy = fig.add_subplot(gs[1, 1])
        ax_entropy.set_facecolor('black')
        
        days_so_far = range(min(day + 1, len(spatial_entropies)))
        
        ax_entropy.plot(days_so_far, spatial_entropies[:len(days_so_far)], 
                       color='#FF69B4', linewidth=3, label='Espacial')
        ax_entropy.plot(days_so_far, global_entropies[:len(days_so_far)], 
                       color='#00FFFF', linewidth=3, label='Global')

        if day < len(spatial_entropies):
            ax_entropy.scatter([day], [spatial_entropies[day]], 
                             color='#FF69B4', s=100, zorder=5)
            ax_entropy.scatter([day], [global_entropies[day]], 
                             color='#00FFFF', s=100, zorder=5)
        
        ax_entropy.set_title('📈 ENTROPIA', fontweight='bold', color='white', fontsize=14)
        ax_entropy.set_xlabel('Dias', color='white')
        ax_entropy.set_ylabel('Entropia (bits)', color='white')
        ax_entropy.legend(loc='upper right')
        ax_entropy.tick_params(colors='white')
        ax_entropy.grid(True, alpha=0.3, color='white')
        
        for spine in ax_entropy.spines.values():
            spine.set_color('white')

        ax_evolution = fig.add_subplot(gs[2, :])
        ax_evolution.set_facecolor('black')
        
        evolution_so_far = history[:day+1, :]
        im_evo = ax_evolution.imshow(evolution_so_far, cmap=cmap, norm=norm, 
                                   aspect='auto', interpolation='none')
        ax_evolution.axhline(y=day, color='yellow', linewidth=3, linestyle='--', alpha=0.8)
        
        ax_evolution.set_title('⏱️ EVOLUÇÃO TEMPORAL', fontweight='bold', color='white', fontsize=14)
        ax_evolution.set_xlabel('Posição na Grade', color='white')
        ax_evolution.set_ylabel('Tempo (dias)', color='white')
        ax_evolution.tick_params(colors='white')
        
        for spine in ax_evolution.spines.values():
            spine.set_color('white')
        
        cbar = plt.colorbar(im_main, ax=[ax_main], orientation='horizontal', 
                           shrink=0.8, pad=0.02, aspect=30)
        cbar.set_ticks([0, 1, 2, 3])
        cbar.set_ticklabels(['🔵 Suscetível', '🟡 Exposto', '🔴 Infectado', '🟢 Recuperado'])
        cbar.ax.tick_params(colors='white', labelsize=12)
        plt.tight_layout()

        frame_filename = os.path.join(gif_frames_dir, f"frame_{frame_count:04d}.png")
        plt.savefig(frame_filename, dpi=150, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        plt.close()
        
        frame_count += 1
    
    return gif_frames_dir, frame_count

def create_spectacular_gif(gif_frames_dir, frame_count, output_dir):
    
    # Carregar todos os frames
    frame_files = []
    for i in range(frame_count):
        frame_path = os.path.join(gif_frames_dir, f"frame_{i:04d}.png")
        if os.path.exists(frame_path):
            frame_files.append(frame_path)
    
    if not frame_files:
        return
    frames = []
    for frame_file in frame_files:
        img = Image.open(frame_file)
        frames.append(img)
    
    gif_path = os.path.join(output_dir, "epidemia_seir_spectacular.gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
        optimize=True
    )

    gif_fast_path = os.path.join(output_dir, "epidemia_seir_fast.gif")
    frames[0].save(
        gif_fast_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        optimize=True
    )
    
    return gif_path, gif_fast_path

def main():
    N = 100000
    E0 = 25
    I0 = 5
    R0_param = 0
    beta = 0.5
    gamma = 0.08
    sigma = 0.125
    total_time = 150
    time_steps = 150
    
    L = 800
    sigma_ca = 0.35
    gamma_ca = 0.06
    block_size = 8
    output_dir = "data_output_seir"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    t, S, E, I, R = run_seir_model(N, E0, I0, R0_param, beta, gamma, sigma, total_time, time_steps)
    ca_history = run_ca_simulation_seir(L, time_steps, sigma_ca, gamma_ca)
    spatial_entropies = [calculate_spatial_entropy(ca_history[i, :], block_size) for i in range(time_steps)]
    global_entropies = [calculate_global_entropy(ca_history[i, :]) for i in range(time_steps)]
    
    plot_seir_curve(t, S, E, I, R, N, output_dir)
    create_perfect_ca_visualization(ca_history, output_dir)
    plot_ca_statistics_perfect(ca_history, t, output_dir)
    plot_entropy_perfect(t, spatial_entropies, global_entropies, output_dir)
    
    gif_frames_dir, frame_count = create_dramatic_gif_frames(ca_history, t, spatial_entropies, global_entropies, output_dir)
    gif_path, gif_fast_path = create_spectacular_gif(gif_frames_dir, frame_count, output_dir)

if __name__ == "__main__":
    main()