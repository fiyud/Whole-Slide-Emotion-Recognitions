def train_model_with_auxiliary_outputs(model, train_loader, optimizer, device, max_grad_norm=1.0):
    model.train()
    total_losses = {
        'total_loss': 0,
        'main_ce_loss': 0,
        'aux_loss': 0,
        'wave_aux_loss': 0,
        'mfcc_aux_loss': 0,
        'intermediate_aux_loss': 0,
        'wasserstein_loss': 0
    }
    
    for batch_idx, (wave, mfcc, label) in enumerate(train_loader):
        wave = wave.unsqueeze(1).float().to(device)
        mfcc = mfcc.unsqueeze(1).float().to(device)
        label = label.long().to(device)

        if torch.isnan(wave).any() or torch.isnan(mfcc).any():
            print(f"NaN detected in inputs at batch {batch_idx}")
            continue

        optimizer.zero_grad()
        
        try:
            loss_dict, aux_outputs = model(wave, mfcc, targets=label, return_loss=True)
            
            total_loss = loss_dict['total_loss']
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"NaN/Inf loss detected at batch {batch_idx}, skipping...")
                continue
                
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            for key in total_losses:
                total_losses[key] += loss_dict[key].item()
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    num_batches = len(train_loader)
    for key in total_losses:
        total_losses[key] /= num_batches
    
    return total_losses

def evaluate_with_auxiliary_outputs(model, test_loader, device, use_ensemble=True):
    model.eval()
    all_labels = []
    all_preds = []
    all_aux_preds = {'wave': [], 'mfcc': [], 'intermediate': [], 'final': []}
    
    with torch.no_grad():
        for wave, mfcc, label in test_loader:
            wave = wave.unsqueeze(1).float().to(device)
            mfcc = mfcc.unsqueeze(1).float().to(device)
            label = label.long().to(device)
            
            aux_outputs = model(wave, mfcc, return_loss=False)
            
            _, wave_pred = torch.max(aux_outputs['wave_aux_logits'], 1)
            _, mfcc_pred = torch.max(aux_outputs['mfcc_aux_logits'], 1)
            _, intermediate_pred = torch.max(aux_outputs['intermediate_aux_logits'], 1)
            _, final_pred = torch.max(aux_outputs['final_logits'], 1)
            
            if use_ensemble:
                # Ensemble prediction (majority voting or averaging logits)
                ensemble_logits = (aux_outputs['wave_aux_logits'] + 
                                 aux_outputs['mfcc_aux_logits'] + 
                                 aux_outputs['intermediate_aux_logits'] + 
                                 aux_outputs['final_logits']) / 4.0
                _, ensemble_pred = torch.max(ensemble_logits, 1)
                all_preds.extend(ensemble_pred.cpu().numpy())
            else:
                # Use only final prediction
                all_preds.extend(final_pred.cpu().numpy())
            
            all_aux_preds['wave'].extend(wave_pred.cpu().numpy())
            all_aux_preds['mfcc'].extend(mfcc_pred.cpu().numpy())
            all_aux_preds['intermediate'].extend(intermediate_pred.cpu().numpy())
            all_aux_preds['final'].extend(final_pred.cpu().numpy())
            
            all_labels.extend(label.cpu().numpy())
    
    return all_labels, all_preds, all_aux_preds
