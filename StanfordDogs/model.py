import timm
import torch
import torch.nn as nn

class StanfordModel(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()

        self.__num_classes = 120
        self.device = device

        self.__backbone = timm.models.vit_base_patch16_224(pretrained=True).to(device)
        for param in self.__backbone.parameters():
            param.requires_grad = False
        
        self.head = nn.Sequential(
            nn.Linear(
            in_features = 1000,
            out_features = 512
            ),

            nn.ReLU(),

            nn.Linear(
            in_features = 512,
            out_features = 256
            ),

            nn.ReLU(),

            nn.Linear(
            in_features = 256,
            out_features = self.__num_classes
            ),

            nn.Softmax()
        ).to(device)

    def __forward(self, x):
        x = self.__backbone(x)
        x = self.head(x)
        return x
    
    def train(self, epoch, dataloader, optimizer=None, criterion=None, version=0):
        if optimizer == None:
            optimizer = torch.optim.Adam(self.head.parameters())
        if criterion == None:
            criterion = torch.nn.CrossEntropyLoss()
        
        for ep in range(epoch):   # 데이터셋을 수차례 반복합니다.

            running_loss = 0.0
            for i, datas in enumerate(dataloader):
                # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
                images, labels = datas
                images = images.to(self.device)  # [100, 3, 224, 224]
                labels = labels.to(self.device)  # [100]

                # 변화도(Gradient) 매개변수를 0으로 만들고
                optimizer.zero_grad()

                # 순전파 + 역전파 + 최적화를 한 후
                outputs = self.__forward(images)
                loss = criterion(outputs.squeeze(), labels.squeeze())
                loss.backward()
                optimizer.step()

                # 통계를 출력합니다.
                running_loss += loss.item()
                
                print(f'[{ep + 1}, {i + 1:5d}] loss: {loss.item():.3f}')

            print(f'[{ep + 1}] loss: {running_loss:.3f}')

        torch.save(self.head.state_dict(), './weights/model_'+str(version)+'.pt')

    def test(self, dataset, criterion=None):
        if criterion == None:
            criterion = torch.nn.CrossEntropyLoss()

        correct_top1 = 0
        loss_sum = 0
        total_cnt = len(dataset)

        with torch.no_grad():
            for i, data in enumerate(dataset):
                # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
                image, label = data
                image = image.to(self.device)  # [100, 3, 224, 224]
                label = label.to(self.device)  # [100]

                # 순전파 + 역전파 + 최적화를 한 후
                output = self.__forward(image[None, ...])
                output = output.squeeze()
                label = label.squeeze()
                loss = criterion(output, label)
                loss_sum += loss

                pred = torch.argmax(output)
                correct_top1 += label[pred]
                
                print(f'[{i + 1:5d}] loss: {loss.item():.3f}')

        print(f'accuracy : {correct_top1/total_cnt}, average loss : {loss_sum/total_cnt}')

    def load_weight(self, path):
        self.head.load_state_dict(torch.load(path))