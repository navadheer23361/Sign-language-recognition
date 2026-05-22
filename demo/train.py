import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# =========================
# CONFIG
# =========================
NUM_CLASSES = 6
CLASS_NAMES = ["Hello","I love you","No","Okay","Please","Thank you"]
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
TEMP = 4.0
ALPHA = 0.7

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================
# SYNTHETIC DATASET
# =========================
class SyntheticDataset(Dataset):
    def __init__(self, size=2000):
        self.size = size
        self.data = []
        self.labels = []

        for i in range(size):
            label = np.random.randint(0, NUM_CLASSES)

            # Create synthetic pattern per class
            img = np.random.randn(3, IMG_SIZE, IMG_SIZE) * 0.2

            img[label % 3] += (label + 1) * 0.3  # pattern

            self.data.append(img.astype(np.float32))
            self.labels.append(label)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), self.labels[idx]

train_loader = DataLoader(SyntheticDataset(2000), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(SyntheticDataset(500), batch_size=BATCH_SIZE)

# =========================
# TEACHER MODEL (STRONG)
# =========================
class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self,x):
        return self.net(x)

# =========================
# STUDENT MODEL (LIGHT)
# =========================
class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self,x):
        return self.net(x)

teacher = TeacherNet().to(DEVICE)
student = StudentNet().to(DEVICE)

# =========================
# LOSS
# =========================
ce = nn.CrossEntropyLoss()
kl = nn.KLDivLoss(reduction='batchmean')

def distill_loss(student_logits, teacher_logits, labels):
    soft_student = nn.functional.log_softmax(student_logits / TEMP, dim=1)
    soft_teacher = nn.functional.softmax(teacher_logits / TEMP, dim=1)

    loss_kd = kl(soft_student, soft_teacher) * (TEMP**2)
    loss_ce = ce(student_logits, labels)

    return ALPHA * loss_kd + (1-ALPHA) * loss_ce

# =========================
# TRAIN TEACHER
# =========================
opt_t = optim.Adam(teacher.parameters(), lr=LR)

teacher_losses = []

print("\n🔥 Training Teacher...\n")

for epoch in range(EPOCHS):
    total = 0
    for x,y in train_loader:
        x,y = x.to(DEVICE), y.to(DEVICE)

        out = teacher(x)
        loss = ce(out,y)

        opt_t.zero_grad()
        loss.backward()
        opt_t.step()

        total += loss.item()

    teacher_losses.append(total)
    print(f"Teacher Epoch {epoch+1}: Loss={total:.3f}")

# =========================
# TRAIN STUDENT (DISTILL)
# =========================
opt_s = optim.Adam(student.parameters(), lr=LR)

student_losses = []

print("\n🔥 Training Student with Distillation...\n")

for epoch in range(EPOCHS):
    total = 0
    for x,y in train_loader:
        x,y = x.to(DEVICE), y.to(DEVICE)

        with torch.no_grad():
            t_out = teacher(x)

        s_out = student(x)

        loss = distill_loss(s_out, t_out, y)

        opt_s.zero_grad()
        loss.backward()
        opt_s.step()

        total += loss.item()

    student_losses.append(total)
    print(f"Student Epoch {epoch+1}: Loss={total:.3f}")

# =========================
# EVALUATION
# =========================
def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)

    return correct/total

teacher_acc = evaluate(teacher)
student_acc = evaluate(student)

print("\n📊 RESULTS")
print(f"Teacher Accuracy: {teacher_acc*100:.2f}%")
print(f"Student Accuracy: {student_acc*100:.2f}%")

# =========================
# PLOT LOSS
# =========================
plt.plot(teacher_losses, label="Teacher Loss")
plt.plot(student_losses, label="Student Loss")
plt.legend()
plt.title("Training Loss")
plt.savefig("loss.png")
plt.show()

# =========================
# SAVE MODELS
# =========================
torch.save(teacher.state_dict(), "teacher.pth")
torch.save(student.state_dict(), "student.pth")