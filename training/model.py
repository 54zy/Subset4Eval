import torch
import torch.nn as nn

class IRT(nn.Module):
    def __init__(self, num_students, num_questions, num_dim):
        super().__init__()
        self.num_dim = num_dim
        self.num_students = num_students
        self.num_questions = num_questions
        self.theta = nn.Embedding(self.num_students, self.num_dim)
        self.beta = nn.Embedding(self.num_questions, 1)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, student_ids, question_ids):
        theta = self.theta(student_ids)
        beta = self.beta(question_ids)
        pred = (theta).sum(dim=1, keepdim=True) - beta
        pred = torch.sigmoid(pred)
        return pred

    def adaptest_save(self, path):
        model_dict = self.state_dict()
        model_dict = {k:v for k,v in model_dict.items() if 'theta' in k or 'beta' in k}
        torch.save(model_dict, path)
