class DualStreamWithAuxiliaryOutputs(nn.Module):
    def __init__(self, lambda_w=0.01, lambda_aux=0.3, num_classes=5):
        super(DualStreamWithAuxiliaryOutputs, self).__init__()
        self.lambda_w = lambda_w
        self.lambda_aux = lambda_aux
        self.num_classes = num_classes

        self.waveform_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),

            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),

            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )

        self.waveform_lstm = nn.GRU(input_size=128, hidden_size=256, batch_first=True, dropout=0.1, bidirectional=True)
        self.waveform_fc = nn.Linear(512, 128)

        self.waveform_aux_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, self.num_classes)
        )

        self.mfcc_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4)),

            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 1),
            nn.Sigmoid()
        )
        
        self.mfcc_gru = nn.GRU(input_size=128, hidden_size=256, batch_first=True, dropout=0.1)
        self.mfcc_fc = nn.Linear(256, 128)

        self.mfcc_aux_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, self.num_classes)
        )

        self.intermediate_aux_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, self.num_classes)
        )

        self.fusion_fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, waveform, mfcc, targets=None, return_loss=False):
        batch_size = waveform.size(0)
        
        # Stream 1: Process waveform (bidirectional GRU)
        try:
            wave_features = self.waveform_extractor(waveform)
            wave_features_lstm = wave_features.permute(0, 2, 1)
            wave_features_lstm, _ = self.waveform_lstm(wave_features_lstm)
            wave_features_final = wave_features_lstm[:, -1, :]  # [batch_size, 512]
            wave_features_final = self.waveform_fc(wave_features_final)  # [batch_size, 128]
            
            # Auxiliary output for waveform stream
            wave_aux_logits = self.waveform_aux_classifier(wave_features_final)
            
        except Exception as e:
            print(f"Error in waveform stream: {e}")
            print(f"Waveform shape: {waveform.shape}")
            if 'wave_features' in locals():
                print(f"Wave features after extractor: {wave_features.shape}")
            if 'wave_features_lstm' in locals():
                print(f"Wave features after GRU: {wave_features_lstm.shape}")
            if 'wave_features_final' in locals():
                print(f"Wave features final shape: {wave_features_final.shape}")
            raise

        # Stream 2: Process MFCC (regular GRU)
        try:
            mfcc_features = self.mfcc_extractor(mfcc)
            
            attention_weights = self.attention(mfcc_features)
            mfcc_features_att = mfcc_features * attention_weights
            
            mfcc_pooled = F.adaptive_avg_pool2d(mfcc_features_att, (1, 1))
            mfcc_pooled = mfcc_pooled.view(batch_size, 128, 1).permute(0, 2, 1)
            
            mfcc_features_gru, _ = self.mfcc_gru(mfcc_pooled)
            mfcc_features_final = mfcc_features_gru[:, -1, :]  # [batch_size, 256]
            mfcc_features_final = self.mfcc_fc(mfcc_features_final)  # [batch_size, 128]
            
            # Auxiliary output for MFCC stream
            mfcc_aux_logits = self.mfcc_aux_classifier(mfcc_features_final)
            
        except Exception as e:
            print(f"Error in MFCC stream: {e}")
            print(f"MFCC shape: {mfcc.shape}")
            if 'mfcc_features' in locals():
                print(f"MFCC features after extractor: {mfcc_features.shape}")
            if 'mfcc_pooled' in locals():
                print(f"MFCC pooled shape: {mfcc_pooled.shape}")
            if 'mfcc_features_gru' in locals():
                print(f"MFCC features after GRU: {mfcc_features_gru.shape}")
            if 'mfcc_features_final' in locals():
                print(f"MFCC features final shape: {mfcc_features_final.shape}")
            raise

        fused_features = torch.cat((wave_features_final, mfcc_features_final), dim=1)
        
        # Intermediate auxiliary output
        intermediate_aux_logits = self.intermediate_aux_classifier(fused_features)
        
        # Final output
        final_logits = self.fusion_fc(fused_features)

        if return_loss and targets is not None:
            main_ce_loss = F.cross_entropy(final_logits, targets)
            
            # Auxiliary losses
            wave_aux_loss = F.cross_entropy(wave_aux_logits, targets)
            mfcc_aux_loss = F.cross_entropy(mfcc_aux_logits, targets)
            intermediate_aux_loss = F.cross_entropy(intermediate_aux_logits, targets)
            
            aux_loss = (wave_aux_loss + mfcc_aux_loss + intermediate_aux_loss) / 3.0
            
            wasserstein_loss = torch.tensor(0.0, device=waveform.device)
            try:
                # Ensure both feature maps are 4D
                wave_features_4d = wave_features.unsqueeze(-1) if len(wave_features.shape) == 3 else wave_features
                mfcc_features_4d = mfcc_features_att.unsqueeze(-1) if len(mfcc_features_att.shape) == 3 else mfcc_features_att
                
                # Resize to match dimensions
                if wave_features_4d.shape[2] != mfcc_features_4d.shape[2] or wave_features_4d.shape[3] != mfcc_features_4d.shape[3]:
                    target_size = (min(wave_features_4d.shape[2], mfcc_features_4d.shape[2]), 
                                 min(wave_features_4d.shape[3], mfcc_features_4d.shape[3]))
                    wave_features_4d = F.interpolate(wave_features_4d, size=target_size, mode='bilinear', align_corners=False)
                    mfcc_features_4d = F.interpolate(mfcc_features_4d, size=target_size, mode='bilinear', align_corners=False)
                
                wdist = wasserstein_features_fast(wave_features_4d, mfcc_features_4d, eps=0.1, n_iter=5)
                wasserstein_loss = torch.clamp(wdist.mean(), min=0.0, max=10.0)
                
            except Exception as e:
                print(f"Warning: Wasserstein loss computation failed: {e}")
                wasserstein_loss = torch.tensor(0.0, device=waveform.device)
            
            total_loss = main_ce_loss + self.lambda_aux * aux_loss + self.lambda_w * wasserstein_loss
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"NaN/Inf detected - Main CE: {main_ce_loss.item()}, Aux: {aux_loss.item()}, W: {wasserstein_loss.item()}")
                total_loss = main_ce_loss  # Fall back to main CE loss only
                aux_loss = torch.tensor(0.0, device=waveform.device)
                wasserstein_loss = torch.tensor(0.0, device=waveform.device)
            
            loss_dict = {
                'total_loss': total_loss,
                'main_ce_loss': main_ce_loss,
                'aux_loss': aux_loss,
                'wave_aux_loss': wave_aux_loss,
                'mfcc_aux_loss': mfcc_aux_loss,
                'intermediate_aux_loss': intermediate_aux_loss,
                'wasserstein_loss': wasserstein_loss
            }
            
            aux_outputs = {
                'wave_aux_logits': wave_aux_logits,
                'mfcc_aux_logits': mfcc_aux_logits,
                'intermediate_aux_logits': intermediate_aux_logits,
                'final_logits': final_logits
            }
            
            return loss_dict, aux_outputs
        
        aux_outputs = {
            'wave_aux_logits': wave_aux_logits,
            'mfcc_aux_logits': mfcc_aux_logits,
            'intermediate_aux_logits': intermediate_aux_logits,
            'final_logits': final_logits
        }
        
        return aux_outputs