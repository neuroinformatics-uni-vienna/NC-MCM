"""
Microvariable Evaluation for Two-Arm Bandit Task

Adapted from: https://github.com/akshey-kumar/comparison-algorithms/blob/6f388b3699a3db0c601a39be314ab89394ce1ba7/evaluation_scripts/microvariable_evaluation.py
Original author: Akshey Kumar
Adapted by: Kerim Atak
"""

import sys
sys.path.append(r'../../../')
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from ncmcm.data_loaders.bandit_task import BanditTaskNeuroPixelsDataset
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split

# Get session directory from command line argument
data_path = sys.argv[1]  # e.g., 'JPAS_0023_20230922'
session_dir = os.path.basename(data_path.rstrip('/'))
# Loading data from two-arm bandit task

print(f"Loading data from: {data_path}")

data = BanditTaskNeuroPixelsDataset(
    data_path=data_path,
    downsample_fs=30,  # Downsample to 30 Hz
    downsample_method='count',
    good_neurons_only=False,
    normalize_method='minmax_global'
)

# Extract neuronal and behavioral data (handle sparse matrices)
X = data.x.toarray().T  # (timepoints, neurons)
B = data.b.toarray().flatten()  # (timepoints,)
print(f"Data shape: X={X.shape}, B={B.shape}")
print(f"Number of behavioral states: {len(np.unique(B))}")

# Get behavior labels from dataset
b_labels_dict = data.b_labels_dict  # Dict mapping state ID to state name
b_labels = data.b_labels  # List of state names ordered by ID
print(f"Behavioral labels: {b_labels_dict}")

# Prepare data with sliding windows
X_, B_ = prep_data(X, B, win=50)
X_train, X_val, B_train_1, B_val_1 = timeseries_train_test_split(X_, B_)
X1_tr = X_train[:,1,:,:]  # Use second window (next state)
X1_val = X_val[:,1,:,:]
print(f"Train shape: {X1_tr.shape}, Validation shape: {X1_val.shape}")

### Analyze label distributions
print("\n" + "="*60)
print("LABEL DISTRIBUTION ANALYSIS")
print("="*60)

# Count labels in train and validation sets
train_label_counts = {}
val_label_counts = {}
for label in np.unique(np.concatenate([B_train_1, B_val_1])):
	train_label_counts[label] = np.sum(B_train_1 == label)
	val_label_counts[label] = np.sum(B_val_1 == label)

total_train = len(B_train_1)
total_val = len(B_val_1)

print(f"\nTotal samples: Train={total_train}, Validation={total_val}")
print(f"\n{'State':<20} {'Train Count':>12} {'Train %':>10} {'Validation Count':>18} {'Validation %':>14}")
print("-" * 70)
for label in sorted(train_label_counts.keys()):
	state_name = b_labels_dict.get(label, f'State {label}')
	train_count = train_label_counts.get(label, 0)
	val_count = val_label_counts.get(label, 0)
	train_pct = 100 * train_count / total_train
	val_pct = 100 * val_count / total_val
	print(f"{state_name:<20} {train_count:>12} {train_pct:>9.1f}% {val_count:>18} {val_pct:>13.1f}%")

# Calculate class imbalance ratio
max_count = max(train_label_counts.values())
min_count = min(train_label_counts.values())
imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
print(f"\nClass imbalance ratio (max/min): {imbalance_ratio:.1f}x")

# Compute class weights for weighted loss (inverse frequency)
num_states = len(np.unique(B_train_1))
state_labels = np.unique(B_train_1)
class_weights = {}
for label in sorted(state_labels):
	# Inverse frequency weighting: weight = total_samples / (num_classes * class_count)
	class_weights[label] = total_train / (num_states * train_label_counts[label]) if train_label_counts[label] > 0 else 1.0

print(f"\nClass weights (inverse frequency):")
for label in sorted(class_weights.keys()):
	state_name = b_labels_dict.get(label, f'State {label}')
	print(f"  {state_name}: {class_weights[label]:.3f}")
print("="*60)

# Create a class weights visualization
fig_weights, ax_weights = plt.subplots(figsize=(10, 6))
weight_values = [class_weights[label] for label in sorted(class_weights.keys())]
x_pos_weights = np.arange(len(state_labels))
bars = ax_weights.bar(x_pos_weights, weight_values, color='steelblue', alpha=0.8, edgecolor='black')

# Color bars based on weight value (higher = more emphasis on underrepresented)
max_weight = max(weight_values)
min_weight = min(weight_values)
for bar, w in zip(bars, weight_values):
	# Color from green (low weight, overrepresented) to red (high weight, underrepresented)
	normalized = (w - min_weight) / (max_weight - min_weight) if max_weight != min_weight else 0.5
	bar.set_color(plt.cm.RdYlGn_r(normalized))

ax_weights.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Baseline (weight=1.0)')
ax_weights.set_xlabel('Behavioral State', fontsize=12)
ax_weights.set_ylabel('Class Weight', fontsize=12)
ax_weights.set_title('Class Weights for Weighted Loss\n(Higher = underrepresented, more emphasis)', fontsize=14, fontweight='bold')
ax_weights.set_xticks(x_pos_weights)
ax_weights.set_xticklabels([b_labels_dict.get(s, f'S{s}') for s in state_labels], rotation=45, ha='right')
ax_weights.legend()
ax_weights.grid(True, alpha=0.3, axis='y')

# Add weight values on top of bars
for i, (bar, w) in enumerate(zip(bars, weight_values)):
	ax_weights.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
					f'{w:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()

# Create output directory early to save weight plot
output_dir = os.path.join(data_path, 'microvariable_evaluation')
os.makedirs(output_dir, exist_ok=True)

plt.savefig(os.path.join(output_dir, f'class_weights_{session_dir}.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, f'class_weights_{session_dir}.pdf'), bbox_inches='tight')
print(f"Saved class weights plot to {output_dir}/")
plt.close()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Convert to PyTorch tensors
X1_tr_tensor = torch.FloatTensor(X1_tr).to(device)
X1_val_tensor = torch.FloatTensor(X1_val).to(device)
B_train_tensor = torch.LongTensor(B_train_1).to(device)
B_val_tensor = torch.LongTensor(B_val_1).to(device)

# Create class weights tensor for weighted loss
weights_list = [class_weights[label] for label in sorted(class_weights.keys())]
class_weights_tensor = torch.FloatTensor(weights_list).to(device)


def train_and_evaluate_decoders(use_weighted_loss=False, suffix=""):
	"""
	Train decoders and evaluate performance.
	
	Args:
		use_weighted_loss: If True, use weighted CrossEntropyLoss for class imbalance
		suffix: String suffix to add to output files (e.g., "_weighted" or "_unweighted")
	
	Returns:
		Dictionary containing all results
	"""
	loss_type = "WEIGHTED" if use_weighted_loss else "UNWEIGHTED"
	print(f"\n{'#'*70}")
	print(f"### TRAINING WITH {loss_type} LOSS ###")
	print(f"{'#'*70}")
	
	print(f"\nTraining 10 decoders with {num_states} behavioral states...")
	print(f"State labels: {state_labels}")
	
	if use_weighted_loss:
		print(f"Using weighted CrossEntropyLoss with weights: {[f'{w:.2f}' for w in weights_list]}")
	else:
		print("Using standard CrossEntropyLoss (no class weights)")
	
	val_acc_list = []
	val_all_predictions = []
	val_all_f1_scores = []

	train_acc_list = []
	train_all_predictions = []
	train_all_f1_scores = []
	
	for i in tqdm(np.arange(10), desc=f'Training decoders ({loss_type})'):
		# Define model
		input_dim = X1_tr.shape[1] * X1_tr.shape[2]
		b_predictor = nn.Sequential(
			nn.Flatten(),
			nn.Linear(input_dim, num_states)
		).to(device)
		
		# Define optimizer and loss
		optimizer = optim.Adam(b_predictor.parameters(), lr=0.01)
		if use_weighted_loss:
			criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
		else:
			criterion = nn.CrossEntropyLoss()
		
		# Training loop
		for epoch in range(100):
			b_predictor.train()
			optimizer.zero_grad()
			outputs = b_predictor(X1_tr_tensor)
			loss = criterion(outputs, B_train_tensor)
			loss.backward()
			optimizer.step()
		
		# Evaluation
		b_predictor.eval()
		with torch.no_grad():
			B1_val_pred = b_predictor(X1_val_tensor).argmax(dim=1).cpu().numpy()
			B1_tr_pred = b_predictor(X1_tr_tensor).argmax(dim=1).cpu().numpy()

		# Validation metrics
		acc = accuracy_score(B_val_1, B1_val_pred)
		val_acc_list.append(acc)
		val_all_predictions.append(B1_val_pred)
		f1_per_class = f1_score(B_val_1, B1_val_pred, average=None, labels=state_labels, zero_division=0)
		val_all_f1_scores.append(f1_per_class)
		
		# Training metrics
		train_acc = accuracy_score(B_train_1, B1_tr_pred)
		train_acc_list.append(train_acc)
		train_all_predictions.append(B1_tr_pred)
		train_f1_per_class = f1_score(B_train_1, B1_tr_pred, average=None, labels=state_labels, zero_division=0)
		train_all_f1_scores.append(train_f1_per_class)

	# Convert to arrays
	val_acc_list = np.array(val_acc_list)
	val_all_predictions = np.array(val_all_predictions)
	val_all_f1_scores = np.array(val_all_f1_scores)
	train_acc_list = np.array(train_acc_list)
	train_all_predictions = np.array(train_all_predictions)
	train_all_f1_scores = np.array(train_all_f1_scores)

	# Print results
	print(f"\n{'='*60}")
	print(f"{loss_type} LOSS - VALIDATION DATA RESULTS")
	print(f"Overall accuracy: {val_acc_list.mean():.3f} ± {val_acc_list.std():.3f}")
	print(f"Median accuracy: {np.median(val_acc_list):.3f}")
	print(f"Min/Max accuracy: {val_acc_list.min():.3f} / {val_acc_list.max():.3f}")

	print(f"\n{loss_type} LOSS - TRAIN DATA RESULTS")
	print(f"Overall accuracy: {train_acc_list.mean():.3f} ± {train_acc_list.std():.3f}")
	print(f"Median accuracy: {np.median(train_acc_list):.3f}")
	print(f"Min/Max accuracy: {train_acc_list.min():.3f} / {train_acc_list.max():.3f}")

	print(f"\n{loss_type} LOSS - OVERFITTING ANALYSIS")
	print(f"Train-Validation gap: {(train_acc_list.mean() - val_acc_list.mean()):.3f}")
	print(f"{'='*60}")

	return {
		'val_acc_list': val_acc_list,
		'val_all_predictions': val_all_predictions,
		'val_all_f1_scores': val_all_f1_scores,
		'train_acc_list': train_acc_list,
		'train_all_predictions': train_all_predictions,
		'train_all_f1_scores': train_all_f1_scores,
		'suffix': suffix,
		'loss_type': loss_type
	}


def save_results_and_visualizations(results):
	"""Save all results and create visualizations for a given training run."""
	
	suffix = results['suffix']
	loss_type = results['loss_type']
	val_acc_list = results['val_acc_list']
	val_all_predictions = results['val_all_predictions']
	val_all_f1_scores = results['val_all_f1_scores']
	train_acc_list = results['train_acc_list']
	train_all_predictions = results['train_all_predictions']
	train_all_f1_scores = results['train_all_f1_scores']
	
	print(f"\nGenerating visualizations for {loss_type} loss...")
	
	# Create output directory
	output_dir = os.path.join(data_path, 'microvariable_evaluation')
	os.makedirs(output_dir, exist_ok=True)
	
	# Save raw results (validation)
	np.savetxt(os.path.join(output_dir, f'acc_list_val_{session_dir}{suffix}.txt'), val_acc_list)
	np.save(os.path.join(output_dir, f'all_predictions_val_{session_dir}{suffix}.npy'), val_all_predictions)
	np.save(os.path.join(output_dir, f'all_f1_scores_val_{session_dir}{suffix}.npy'), val_all_f1_scores)
	
	# Save raw results (train)
	np.savetxt(os.path.join(output_dir, f'acc_list_train_{session_dir}{suffix}.txt'), train_acc_list)
	np.save(os.path.join(output_dir, f'all_predictions_train_{session_dir}{suffix}.npy'), train_all_predictions)
	np.save(os.path.join(output_dir, f'all_f1_scores_train_{session_dir}{suffix}.npy'), train_all_f1_scores)
	
	# Pre-compute values
	x_pos = np.arange(len(state_labels))
	val_f1_means = val_all_f1_scores.mean(axis=0)
	train_f1_means = train_all_f1_scores.mean(axis=0)
	f1_gap = train_f1_means - val_f1_means
	
	# Compute average confusion matrices
	avg_conf_matrix = np.zeros((num_states, num_states))
	for pred in val_all_predictions:
		avg_conf_matrix += confusion_matrix(B_val_1, pred, labels=state_labels)
	avg_conf_matrix /= len(val_all_predictions)
	
	train_avg_conf_matrix = np.zeros((num_states, num_states))
	for pred in train_all_predictions:
		train_avg_conf_matrix += confusion_matrix(B_train_1, pred, labels=state_labels)
	train_avg_conf_matrix /= len(train_all_predictions)
	
	### Create comprehensive train/validation comparison visualizations
	fig = plt.figure(figsize=(24, 16))
	fig.suptitle(f'{loss_type} Loss Results', fontsize=18, fontweight='bold', y=1.02)
	
	# 0. Label distribution (Train vs Validation)
	ax0 = plt.subplot(2, 4, 1)
	train_counts = [train_label_counts.get(s, 0) for s in state_labels]
	val_counts = [val_label_counts.get(s, 0) for s in state_labels]
	width = 0.35
	ax0.bar(x_pos - width/2, train_counts, width, label='Train', color='lightgreen', alpha=0.8)
	ax0.bar(x_pos + width/2, val_counts, width, label='Validation', color='skyblue', alpha=0.8)
	ax0.set_xlabel('Behavioral State', fontsize=12)
	ax0.set_ylabel('Sample Count', fontsize=12)
	ax0.set_title('Label Distribution (Train vs Validation)', fontsize=14, fontweight='bold')
	ax0.set_xticks(x_pos)
	ax0.set_xticklabels([b_labels_dict.get(s, f'S{s}') for s in state_labels], rotation=45, ha='right')
	ax0.legend()
	ax0.grid(True, alpha=0.3, axis='y')
	for i, (tc, vc) in enumerate(zip(train_counts, val_counts)):
		ax0.annotate(f'{100*tc/total_train:.1f}%', (i - width/2, tc), ha='center', va='bottom', fontsize=8)
		ax0.annotate(f'{100*vc/total_val:.1f}%', (i + width/2, vc), ha='center', va='bottom', fontsize=8)
	
	# 1. Train vs Validation accuracy comparison boxplot
	ax1 = plt.subplot(2, 4, 2)
	bp1 = ax1.boxplot([train_acc_list, val_acc_list], positions=[1, 2], widths=0.6,
					   patch_artist=True, showmeans=True,
					   meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
	colors_train_val = ['lightgreen', 'skyblue']
	for patch, color in zip(bp1['boxes'], colors_train_val):
		patch.set_facecolor(color)
	ax1.set_xticklabels(['Train', 'Validation'])
	ax1.set_ylabel('Accuracy', fontsize=12)
	ax1.set_title(f'Train vs Validation Accuracy\nGap: {(train_acc_list.mean() - val_acc_list.mean()):.3f}', fontsize=14, fontweight='bold')
	ax1.grid(True, alpha=0.3, axis='y')
	ax1.set_ylim([0, 1.05])
	
	# 2. Per-state F1 score boxplots (Train vs Validation)
	ax2 = plt.subplot(2, 4, 3)
	f1_df_data = []
	for state_idx, state_label in enumerate(state_labels):
		for run_idx in range(val_all_f1_scores.shape[0]):
			f1_df_data.append({
				'State': b_labels_dict.get(state_label, f'State {state_label}'),
				'F1 Score': val_all_f1_scores[run_idx, state_idx],
				'Dataset': 'Validation'
			})
			f1_df_data.append({
				'State': b_labels_dict.get(state_label, f'State {state_label}'),
				'F1 Score': train_all_f1_scores[run_idx, state_idx],
				'Dataset': 'Train'
			})
	f1_df = pd.DataFrame(f1_df_data)
	sns.boxplot(data=f1_df, x='State', y='F1 Score', hue='Dataset', ax=ax2, palette={'Train': 'lightgreen', 'Validation': 'skyblue'})
	ax2.set_xlabel('Behavioral State', fontsize=12)
	ax2.set_ylabel('F1 Score', fontsize=12)
	ax2.set_title('Per-State F1 Score (Train vs Validation)', fontsize=14, fontweight='bold')
	ax2.tick_params(axis='x', rotation=45)
	ax2.grid(True, alpha=0.3, axis='y')
	ax2.legend(title='Dataset')
	
	# 3. Average confusion matrix (VALIDATION)
	ax3 = plt.subplot(2, 4, 4)
	sns.heatmap(avg_conf_matrix, annot=True, fmt='.1f', cmap='Blues', ax=ax3,
				xticklabels=[b_labels_dict.get(s, f'S{s}') for s in state_labels],
				yticklabels=[b_labels_dict.get(s, f'S{s}') for s in state_labels],
				cbar_kws={'label': 'Average Count'})
	ax3.set_xlabel('Predicted State', fontsize=12)
	ax3.set_ylabel('True State', fontsize=12)
	ax3.set_title('Average Confusion Matrix (VALIDATION)', fontsize=14, fontweight='bold')
	
	# 4. Average confusion matrix (TRAIN)
	ax4 = plt.subplot(2, 4, 5)
	sns.heatmap(train_avg_conf_matrix, annot=True, fmt='.1f', cmap='Greens', ax=ax4,
				xticklabels=[b_labels_dict.get(s, f'S{s}') for s in state_labels],
				yticklabels=[b_labels_dict.get(s, f'S{s}') for s in state_labels],
				cbar_kws={'label': 'Average Count'})
	ax4.set_xlabel('Predicted State', fontsize=12)
	ax4.set_ylabel('True State', fontsize=12)
	ax4.set_title('Average Confusion Matrix (TRAIN)', fontsize=14, fontweight='bold')
	
	# 5. Per-state F1 comparison (Train vs Validation bar chart)
	ax5 = plt.subplot(2, 4, 6)
	width = 0.35
	ax5.bar(x_pos - width/2, train_f1_means, width, label='Train', color='lightgreen', alpha=0.8)
	ax5.bar(x_pos + width/2, val_f1_means, width, label='Validation', color='skyblue', alpha=0.8)
	ax5.set_xlabel('Behavioral State', fontsize=12)
	ax5.set_ylabel('F1 Score', fontsize=12)
	ax5.set_title('Per-State F1: Train vs Validation', fontsize=14, fontweight='bold')
	ax5.set_xticks(x_pos)
	ax5.set_xticklabels([b_labels_dict.get(s, f'S{s}') for s in state_labels], rotation=45, ha='right')
	ax5.legend()
	ax5.grid(True, alpha=0.3, axis='y')
	ax5.set_ylim([0, 1.05])
	
	# 6. Train-Validation F1 gap per state
	ax6 = plt.subplot(2, 4, 7)
	colors_gap = ['green' if g > 0 else 'red' for g in f1_gap]
	ax6.bar(x_pos, f1_gap, color=colors_gap, alpha=0.7)
	ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
	ax6.axhline(y=f1_gap.mean(), color='red', linestyle='--', label=f'Mean gap: {f1_gap.mean():.3f}')
	ax6.set_xlabel('Behavioral State', fontsize=12)
	ax6.set_ylabel('F1 Gap (Train - Validation)', fontsize=12)
	ax6.set_title('Overfitting Analysis: F1 Gap per State', fontsize=14, fontweight='bold')
	ax6.set_xticks(x_pos)
	ax6.set_xticklabels([b_labels_dict.get(s, f'S{s}') for s in state_labels], rotation=45, ha='right')
	ax6.legend()
	ax6.grid(True, alpha=0.3, axis='y')
	
	# 7. F1 Score vs Sample Count
	ax7 = plt.subplot(2, 4, 8)
	val_counts_arr = np.array([val_label_counts.get(s, 0) for s in state_labels])
	ax7.scatter(val_counts_arr, val_f1_means, s=100, c='skyblue', edgecolors='blue', alpha=0.8)
	for i, (x, y) in enumerate(zip(val_counts_arr, val_f1_means)):
		ax7.annotate(b_labels_dict.get(state_labels[i], f'S{state_labels[i]}'), 
					 (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
	ax7.set_xlabel('Validation Sample Count', fontsize=12)
	ax7.set_ylabel('Validation F1 Score', fontsize=12)
	ax7.set_title('F1 Score vs Sample Count\n(Class Imbalance Effect)', fontsize=14, fontweight='bold')
	ax7.grid(True, alpha=0.3)
	corr = np.corrcoef(val_counts_arr, val_f1_means)[0, 1]
	ax7.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax7.transAxes, 
			 verticalalignment='top', fontsize=11,
			 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
	
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, f'train_validation_comparison_{session_dir}{suffix}.png'), dpi=300, bbox_inches='tight')
	plt.savefig(os.path.join(output_dir, f'train_validation_comparison_{session_dir}{suffix}.pdf'), bbox_inches='tight')
	print(f"Saved train/validation comparison plots ({loss_type})")
	
	### Create validation-only detailed visualizations
	fig_val = plt.figure(figsize=(20, 12))
	fig_val.suptitle(f'{loss_type} Loss - Detailed Analysis', fontsize=16, fontweight='bold', y=1.02)
	
	# 1. Overall accuracy distribution boxplot (validation)
	ax1t = plt.subplot(2, 3, 1)
	sns.boxplot(y=val_acc_list, ax=ax1t, color='skyblue')
	ax1t.axhline(y=val_acc_list.mean(), color='red', linestyle='--', label=f'Mean: {val_acc_list.mean():.3f}')
	ax1t.set_ylabel('Accuracy', fontsize=12)
	ax1t.set_title('Overall Decoder Accuracy Distribution\n(10 runs, validation)', fontsize=14, fontweight='bold')
	ax1t.legend()
	ax1t.grid(True, alpha=0.3)
	
	# 2. Per-state F1 score boxplots
	ax2t = plt.subplot(2, 3, 2)
	f1_df_val_only = []
	for state_idx, state_label in enumerate(state_labels):
		for run_idx in range(val_all_f1_scores.shape[0]):
			f1_df_val_only.append({
				'State': b_labels_dict.get(state_label, f'State {state_label}'),
				'F1 Score': val_all_f1_scores[run_idx, state_idx]
			})
	f1_df_val = pd.DataFrame(f1_df_val_only)
	sns.boxplot(data=f1_df_val, x='State', y='F1 Score', ax=ax2t, palette='Set2')
	ax2t.set_xlabel('Behavioral State', fontsize=12)
	ax2t.set_ylabel('F1 Score', fontsize=12)
	ax2t.set_title('Per-State F1 Score Distribution', fontsize=14, fontweight='bold')
	ax2t.tick_params(axis='x', rotation=45)
	ax2t.grid(True, alpha=0.3, axis='y')
	
	# 3. Average confusion matrix
	ax3t = plt.subplot(2, 3, 3)
	sns.heatmap(avg_conf_matrix, annot=True, fmt='.1f', cmap='Blues', ax=ax3t,
				xticklabels=[b_labels_dict.get(s, f'S{s}') for s in state_labels],
				yticklabels=[b_labels_dict.get(s, f'S{s}') for s in state_labels],
				cbar_kws={'label': 'Average Count'})
	ax3t.set_xlabel('Predicted State', fontsize=12)
	ax3t.set_ylabel('True State', fontsize=12)
	ax3t.set_title('Average Confusion Matrix', fontsize=14, fontweight='bold')
	
	# 4. Normalized confusion matrix
	ax4t = plt.subplot(2, 3, 4)
	norm_conf_matrix = avg_conf_matrix / avg_conf_matrix.sum(axis=1, keepdims=True) * 100
	sns.heatmap(norm_conf_matrix, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax4t,
				xticklabels=[b_labels_dict.get(s, f'S{s}') for s in state_labels],
				yticklabels=[b_labels_dict.get(s, f'S{s}') for s in state_labels],
				cbar_kws={'label': 'Percentage (%)'})
	ax4t.set_xlabel('Predicted State', fontsize=12)
	ax4t.set_ylabel('True State', fontsize=12)
	ax4t.set_title('Normalized Confusion Matrix (%)', fontsize=14, fontweight='bold')
	
	# 5. Per-state metrics (Precision, Recall, F1)
	ax5t = plt.subplot(2, 3, 5)
	avg_precision = []
	avg_recall = []
	avg_f1 = []
	for state_idx, state_label in enumerate(state_labels):
		state_precisions = []
		state_recalls = []
		for pred in val_all_predictions:
			prec = precision_score(B_val_1, pred, labels=[state_label], average='macro', zero_division=0)
			rec = recall_score(B_val_1, pred, labels=[state_label], average='macro', zero_division=0)
			state_precisions.append(prec)
			state_recalls.append(rec)
		avg_precision.append(np.mean(state_precisions))
		avg_recall.append(np.mean(state_recalls))
		avg_f1.append(np.mean(val_all_f1_scores[:, state_idx]))
	
	width = 0.25
	ax5t.bar(x_pos - width, avg_precision, width, label='Precision', alpha=0.8)
	ax5t.bar(x_pos, avg_recall, width, label='Recall', alpha=0.8)
	ax5t.bar(x_pos + width, avg_f1, width, label='F1 Score', alpha=0.8)
	ax5t.set_xlabel('Behavioral State', fontsize=12)
	ax5t.set_ylabel('Score', fontsize=12)
	ax5t.set_title('Average Precision, Recall, F1 per State', fontsize=14, fontweight='bold')
	ax5t.set_xticks(x_pos)
	ax5t.set_xticklabels([b_labels_dict.get(s, f'S{s}') for s in state_labels], rotation=45, ha='right')
	ax5t.legend()
	ax5t.grid(True, alpha=0.3, axis='y')
	ax5t.set_ylim([0, 1.05])
	
	# 6. State-wise F1 with error bars
	ax6t = plt.subplot(2, 3, 6)
	val_f1_stds = val_all_f1_scores.std(axis=0)
	ax6t.errorbar(state_labels, val_f1_means, yerr=val_f1_stds, 
				  fmt='o-', capsize=5, capthick=2, markersize=8, linewidth=2)
	ax6t.set_xlabel('Behavioral State', fontsize=12)
	ax6t.set_ylabel('F1 Score', fontsize=12)
	ax6t.set_title('F1 Score per State (mean ± std)', fontsize=14, fontweight='bold')
	ax6t.grid(True, alpha=0.3)
	ax6t.set_ylim([0, 1.05])
	
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, f'detailed_analysis_{session_dir}{suffix}.png'), dpi=300, bbox_inches='tight')
	plt.savefig(os.path.join(output_dir, f'detailed_analysis_{session_dir}{suffix}.pdf'), bbox_inches='tight')
	print(f"Saved detailed analysis plots ({loss_type})")
	
	# Print detailed text report
	state_f1_stds = val_all_f1_scores.std(axis=0)
	train_f1_stds = train_all_f1_scores.std(axis=0)
	
	print(f"\n{'='*60}")
	print(f"{loss_type} LOSS - DETAILED PERFORMANCE REPORT (PER STATE)")
	print("="*60)
	for state_idx, state_label in enumerate(state_labels):
		state_name = b_labels_dict.get(state_label, f'State {state_label}')
		print(f"\n{state_name} (ID: {state_label}):")
		print(f"  Train F1:   {train_f1_means[state_idx]:.3f} ± {train_f1_stds[state_idx]:.3f}")
		print(f"  Validation F1:    {val_f1_means[state_idx]:.3f} ± {state_f1_stds[state_idx]:.3f}")
		print(f"  Gap:        {f1_gap[state_idx]:.3f}")
		train_count = np.sum(B_train_1 == state_label)
		val_count = np.sum(B_val_1 == state_label)
		print(f"  Train/Validation samples: {train_count} / {val_count}")
	print("\n" + "="*60)
	
	plt.close('all')
	return output_dir


### Run UNWEIGHTED loss training first
results_unweighted = train_and_evaluate_decoders(use_weighted_loss=False, suffix="_unweighted")
output_dir = save_results_and_visualizations(results_unweighted)

### Run WEIGHTED loss training
results_weighted = train_and_evaluate_decoders(use_weighted_loss=True, suffix="_weighted")
save_results_and_visualizations(results_weighted)

### Create comparison visualization between WEIGHTED and UNWEIGHTED
print("\n" + "#"*70)
print("### COMPARING WEIGHTED VS UNWEIGHTED LOSS ###")
print("#"*70)

fig_compare = plt.figure(figsize=(20, 12))
fig_compare.suptitle('Weighted vs Unweighted Loss Comparison', fontsize=18, fontweight='bold')

x_pos = np.arange(len(state_labels))

# 1. Accuracy comparison
ax1c = plt.subplot(2, 3, 1)
bp = ax1c.boxplot([results_unweighted['val_acc_list'], results_weighted['val_acc_list']], 
			   positions=[1, 2], widths=0.6, patch_artist=True, showmeans=True,
			   meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
colors_compare = ['lightcoral', 'lightgreen']
for patch, color in zip(bp['boxes'], colors_compare):
	patch.set_facecolor(color)
ax1c.set_xticklabels(['Unweighted', 'Weighted'])
ax1c.set_ylabel('Validation Accuracy', fontsize=12)
ax1c.set_title('Validation Accuracy: Weighted vs Unweighted', fontsize=14, fontweight='bold')
ax1c.grid(True, alpha=0.3, axis='y')
ax1c.set_ylim([0, 1.05])

# Add mean values as text
ax1c.text(1, results_unweighted['val_acc_list'].mean() + 0.02, 
		  f'{results_unweighted["val_acc_list"].mean():.3f}', ha='center', fontsize=10)
ax1c.text(2, results_weighted['val_acc_list'].mean() + 0.02, 
		  f'{results_weighted["val_acc_list"].mean():.3f}', ha='center', fontsize=10)

# 2. Per-state F1 comparison
ax2c = plt.subplot(2, 3, 2)
unweighted_f1_means = results_unweighted['val_all_f1_scores'].mean(axis=0)
weighted_f1_means = results_weighted['val_all_f1_scores'].mean(axis=0)
width = 0.35
ax2c.bar(x_pos - width/2, unweighted_f1_means, width, label='Unweighted', color='lightcoral', alpha=0.8)
ax2c.bar(x_pos + width/2, weighted_f1_means, width, label='Weighted', color='lightgreen', alpha=0.8)
ax2c.set_xlabel('Behavioral State', fontsize=12)
ax2c.set_ylabel('Validation F1 Score', fontsize=12)
ax2c.set_title('Per-State Validation F1: Weighted vs Unweighted', fontsize=14, fontweight='bold')
ax2c.set_xticks(x_pos)
ax2c.set_xticklabels([b_labels_dict.get(s, f'S{s}') for s in state_labels], rotation=45, ha='right')
ax2c.legend()
ax2c.grid(True, alpha=0.3, axis='y')
ax2c.set_ylim([0, 1.05])

# 3. F1 improvement from weighting
ax3c = plt.subplot(2, 3, 3)
f1_improvement = weighted_f1_means - unweighted_f1_means
colors_imp = ['green' if imp > 0 else 'red' for imp in f1_improvement]
ax3c.bar(x_pos, f1_improvement, color=colors_imp, alpha=0.7)
ax3c.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3c.axhline(y=f1_improvement.mean(), color='blue', linestyle='--', 
			 label=f'Mean improvement: {f1_improvement.mean():.3f}')
ax3c.set_xlabel('Behavioral State', fontsize=12)
ax3c.set_ylabel('F1 Improvement (Weighted - Unweighted)', fontsize=12)
ax3c.set_title('F1 Score Improvement from Weighted Loss', fontsize=14, fontweight='bold')
ax3c.set_xticks(x_pos)
ax3c.set_xticklabels([b_labels_dict.get(s, f'S{s}') for s in state_labels], rotation=45, ha='right')
ax3c.legend()
ax3c.grid(True, alpha=0.3, axis='y')

# 4. F1 improvement vs class frequency (scatter)
ax4c = plt.subplot(2, 3, 4)
train_counts_arr = np.array([train_label_counts.get(s, 0) for s in state_labels])
ax4c.scatter(train_counts_arr, f1_improvement, s=100, c='purple', edgecolors='black', alpha=0.8)
for i, (x, y) in enumerate(zip(train_counts_arr, f1_improvement)):
	ax4c.annotate(b_labels_dict.get(state_labels[i], f'S{state_labels[i]}'), 
				 (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
ax4c.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax4c.set_xlabel('Training Sample Count', fontsize=12)
ax4c.set_ylabel('F1 Improvement', fontsize=12)
ax4c.set_title('F1 Improvement vs Class Frequency\n(Do underrepresented classes benefit more?)', fontsize=14, fontweight='bold')
ax4c.grid(True, alpha=0.3)

# Add correlation
corr_imp = np.corrcoef(train_counts_arr, f1_improvement)[0, 1]
ax4c.text(0.05, 0.95, f'Correlation: {corr_imp:.3f}', transform=ax4c.transAxes, 
		 verticalalignment='top', fontsize=11,
		 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 5. Per-state F1 boxplot comparison
ax5c = plt.subplot(2, 3, 5)
f1_compare_data = []

for state_idx, state_label in enumerate(state_labels):
	for run_idx in range(results_unweighted['val_all_f1_scores'].shape[0]):
		f1_compare_data.append({
			'State': b_labels_dict.get(state_label, f'State {state_label}'),
			'F1 Score': results_unweighted['val_all_f1_scores'][run_idx, state_idx],
			'Loss Type': 'Unweighted'
		})
		f1_compare_data.append({
			'State': b_labels_dict.get(state_label, f'State {state_label}'),
			'F1 Score': results_weighted['val_all_f1_scores'][run_idx, state_idx],
			'Loss Type': 'Weighted'
		})
f1_compare_df = pd.DataFrame(f1_compare_data)
sns.boxplot(data=f1_compare_df, x='State', y='F1 Score', hue='Loss Type', ax=ax5c, 
			palette={'Unweighted': 'lightcoral', 'Weighted': 'lightgreen'})
ax5c.set_xlabel('Behavioral State', fontsize=12)
ax5c.set_ylabel('Validation F1 Score', fontsize=12)
ax5c.set_title('Per-State Validation F1 Distribution: Weighted vs Unweighted', fontsize=14, fontweight='bold')
ax5c.tick_params(axis='x', rotation=45)
ax5c.grid(True, alpha=0.3, axis='y')

# 6. Summary statistics table
ax6c = plt.subplot(2, 3, 6)
ax6c.axis('off')
summary_text = f"""
SUMMARY COMPARISON
{'='*40}

OVERALL VALIDATION ACCURACY:
	Unweighted: {results_unweighted['val_acc_list'].mean():.3f} ± {results_unweighted['val_acc_list'].std():.3f}
	Weighted:   {results_weighted['val_acc_list'].mean():.3f} ± {results_weighted['val_acc_list'].std():.3f}
	Difference: {results_weighted['val_acc_list'].mean() - results_unweighted['val_acc_list'].mean():+.3f}

MACRO F1 SCORE:
  Unweighted: {unweighted_f1_means.mean():.3f}
  Weighted:   {weighted_f1_means.mean():.3f}
  Difference: {weighted_f1_means.mean() - unweighted_f1_means.mean():+.3f}

PER-STATE F1 IMPROVEMENTS:
"""
for state_idx, state_label in enumerate(state_labels):
	state_name = b_labels_dict.get(state_label, f'State {state_label}')
	summary_text += f"  {state_name}: {f1_improvement[state_idx]:+.3f}\n"

ax6c.text(0.1, 0.95, summary_text, transform=ax6c.transAxes, fontsize=11,
		  verticalalignment='top', family='monospace',
		  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'weighted_vs_unweighted_comparison_{session_dir}.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, f'weighted_vs_unweighted_comparison_{session_dir}.pdf'), bbox_inches='tight')
print(f"Saved weighted vs unweighted comparison to {output_dir}/")

### Estimating the chance accuracy of behaviour decoding
print("\nEstimating chance accuracy...")
chance_acc = np.zeros(500)
for i, _ in enumerate(chance_acc):
	B_perm = np.random.choice(B_val_1, size=B_val_1.shape)
	chance_acc[i] = accuracy_score(B_perm, B_val_1)
print(f'Chance prediction accuracy: {chance_acc.mean():.3f} ± {chance_acc.std():.3f}')
np.savetxt(os.path.join(output_dir, f'acc_list_chance_{session_dir}.txt'), chance_acc)

# Create comparison plot: decoder vs chance (using both weighted and unweighted)
fig2, ax = plt.subplots(1, 1, figsize=(12, 6))
positions = [1, 2, 3]
bp = ax.boxplot([results_unweighted['val_acc_list'], results_weighted['val_acc_list'], chance_acc], 
				positions=positions, widths=0.6,
				patch_artist=True, showmeans=True,
				meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
colors = ['lightcoral', 'lightgreen', 'lightgray']
for patch, color in zip(bp['boxes'], colors):
	patch.set_facecolor(color)
ax.set_xticklabels(['Decoder\n(Unweighted)', 'Decoder\n(Weighted)', 'Chance\n(Random)'])
ax.set_ylabel('Accuracy', fontsize=14)
ax.set_title('Decoder Performance vs Chance Level', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.0])

# Add statistical annotations
t_stat_uw, p_value_uw = stats.ttest_ind(results_unweighted['val_acc_list'], chance_acc)
t_stat_w, p_value_w = stats.ttest_ind(results_weighted['val_acc_list'], chance_acc)
ax.text(0.5, 0.95, f'Unweighted vs Chance: t={t_stat_uw:.2f}, p={p_value_uw:.2e}\nWeighted vs Chance: t={t_stat_w:.2f}, p={p_value_w:.2e}', 
		transform=ax.transAxes, ha='center', va='top',
		bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
		fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'decoder_vs_chance_{session_dir}.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, f'decoder_vs_chance_{session_dir}.pdf'), bbox_inches='tight')
print(f"Saved decoder vs chance comparison to {output_dir}/")

print(f"\n{'='*60}")
print(f"All results saved to {output_dir}/")
print(f"{'='*60}")
plt.close('all')