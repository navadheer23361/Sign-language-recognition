import torch
import torch.nn.functional as F

def distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    temperature=4.0,
    alpha=0.7
):

    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)

    hard_loss = F.cross_entropy(
        student_logits,
        labels
    )

    total_loss = (
        alpha * hard_loss
        +
        (1 - alpha) * soft_loss
    )

    return total_loss