import torch
import torch.nn.functional as F
from trl import SFTTrainer
from peft import PeftModel


class TemaTrainer(SFTTrainer):
    """
    EMA teacher trainer (GPU-only, optimized for LoRA).
    - EMA обновляет только веса LoRA (A, B матрицы).
    - Teacher не хранит всю модель -> только EMA-копию LoRA-параметров.
    - Это сильно экономит память: нет второй копии всей LLM.
    """
    def __init__(self, ema_decay=0.999, alpha=1.0, beta=0.05, temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.ema_decay = float(ema_decay)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.temperature = float(temperature)

        # EMA-хранилище только для LoRA-параметров
        self.ema_state = {}
        self._init_ema()

    def _init_ema(self):
        """Сохраняем только LoRA-параметры"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:  # только LoRA
                self.ema_state[name] = param.detach().clone().float()

    @torch.no_grad()
    def _update_ema(self):
        """EMA-обновление только LoRA параметров."""
        decay = self.ema_decay
        for name, param in self.model.named_parameters():
            if name in self.ema_state:
                ema_param = self.ema_state[name]
                ema_param.mul_(decay).add_(param.detach().float(), alpha=(1.0 - decay))

    def _apply_ema_weights(self):
        """Создаём копию LoRA state_dict с EMA-весами."""
        ema_lora_state = {}
        for name, param in self.model.named_parameters():
            if name in self.ema_state:
                ema_lora_state[name] = self.ema_state[name].clone().to(param.device)
        return ema_lora_state

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        ce_loss = outputs.loss
        student_logits = outputs.logits

        with torch.no_grad():
            ema_state = self._apply_ema_weights()

            # временно подменяем веса модели
            backup = {}
            for name, param in model.named_parameters():
                if name in ema_state:
                    backup[name] = param.data.clone()
                    param.data.copy_(ema_state[name].to(param.device))

            teacher_logits = model(**inputs, use_cache=False).logits

            # откатываем LoRA на студентские веса
            for name, param in model.named_parameters():
                if name in backup:
                    param.data.copy_(backup[name])

        valid_mask = labels.ne(-100)
        student_sel = student_logits[valid_mask].float()
        teacher_sel = teacher_logits[valid_mask].float()

        T = self.temperature
        teacher_logp = F.log_softmax(teacher_sel / T, dim=-1)
        student_logp = F.log_softmax(student_sel / T, dim=-1)

        kl = F.kl_div(student_logp, teacher_logp.exp(), reduction="batchmean") * (T ** 2)

        # total loss
        total_loss = self.alpha * ce_loss + self.beta * kl

        # EMA update
        self._update_ema()

        return (total_loss, outputs) if return_outputs else total_loss