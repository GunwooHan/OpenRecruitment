# 이런 분을 찾습니다!

---
## 팀원에게 바라는 점
- 자료 정리를 잘했으면 좋겠어요! 
- 참여가 적극적이고 의사소통이 활발했으면 좋겠어요!
- 서로를 존중하고 예의를 갖춰줬으면 좋겠어요!
- 지치더라도 끝까지 포기하지 않고 마무리할 수 있으면 좋겠습니다!
- 모르는 점에 대해 부끄러워하지 않고 질문해주면 좋겠어요! 질문이 많아도 괜찮아요!
- git을 잘 몰라도 괜찮아요! 저도 잘 못해서 같이 공부하면서 했으면 좋겠어요!
---
## 우리 조는 이렇게 진행하고 싶어요!

---
### 회의는 이렇게 할거에요!
- 모든 팀원이 각자 공부한 내용이나 새로 알게된 것, 별도로 정리한 내용등을 자유롭게 공유합니다!
- 모든 회의는 수업 내용, 대회 상황에 따라 유동적으로 진행됩니다!
- 팀원으로 합류하시면 앞으로 어떤 방향으로 진행할지에 대해서도 같이 이야기해보면 좋을 것 같아요!


### 코드 작성은 이렇게 할거에요!
- 서로 코드를 이해하기 쉽게 **Pytorch lightning**로 구현할거에요!
    ```python
    if __name__ == '__main__':
        seed_fix(args.seed)
        train_transform, test_transform = make_transform(args)
        dataset = ImageBaseDataset(data_dir='train/train/images', transform=test_transform)
        train_dataset, valid_dataset = dataset.split_dataset()

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
        val_loader = DataLoader(valid_dataset, batch_size=128, num_workers=4)

        # model
        model = CustomModel(model_name=args.model, num_classes=18)

        # training
        trainer = pl.Trainer(max_epochs=args.epochs, gpus=2, accelerator="dp")
        trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

    ```
- train 함수에 누덕누덕 구현하지 않고 **클래스 단위로 개발**해서 사용할거에요!

    ```python
    class FocalLoss(nn.Module):
        def __init__(self, weight=None, gamma=2., reduction='mean'):
            nn.Module.__init__(self)
            self.weight = weight
            self.gamma = gamma
            self.reduction = reduction
        def forward(self, input_tensor, target_tensor):
            log_prob = F.log_softmax(input_tensor, dim=-1)
            prob = torch.exp(log_prob)
            return F.nll_loss(
                ((1 - prob) ** self.gamma) * log_prob,
                target_tensor,
                weight=self.weight,
                reduction=self.reduction)

    -------------------------------------------------------------------------

    class CustomModel(pl.LightningModule):

        def __init__(self, model_name='tf_efficientnet_b0', num_classes=18):
            self.focal_loss = FocalLoss()
        ...

        def training_step(self, train_batch, batch_idx):
            x, y = train_batch
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
            x_hat = self.decoder(z)
            loss = self.focal_loss(x_hat, x)      # Loss 수정
            wandb.log('val_loss', loss)

    ```
- **argparse**를 활용하여 **wandb** hyper parameter tuning 사용해 실험할 거에요!

    ```python
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='base')
    parser.add_argument('--save_path', type=str, default='saved')
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--label_type', type=str, default='all')

    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--test_size', type=float, default=0.2)

    ...

    args = parser.parse_args()
    ```
    - 이전 Competition에는 이만큼 실험했어요!🙆‍♀️
    ![](https://i.imgur.com/YriH5Gi.png)

    
---

## 팀원 소개
- `박범수` : 학부생때 임베디드, C#을 위주로 하다가 프로그래밍을 하다가 이번에 본격적으로 CV에 대해 관심이 생겨 합류하게 됐습니다! 아직 부족한 부분이 많아 서로 협력하면서 배워나가고 싶습니다!
- `한건우` : GAN으로 아이돌을 만들다 부스트캠프에 합류했어요! Segmentation도 조금 할줄 알아요! 같이 성장하면서 부스트캠프가 끝나도 쭉 알고지냈으면 좋겠어요! 
  - 만들던 아이돌 영상 : https://youtu.be/biIVEH7-Rao
  - 논문은 요정도 읽어요!
      - Pyramid Attention Network for Semantic Sementation : https://docs.google.com/presentation/d/15jgE8P26fH_BdZTJTNH6fm0IIRjTzbXk/edit?usp=sharing&ouid=107012802823432471585&rtpof=true&sd=true
      - CartoonGAN : https://docs.google.com/presentation/d/14x06uOQKxgoqbkYZUbXWn_Myenn9o16v/edit?usp=sharing&ouid=107012802823432471585&rtpof=true&sd=true
      - RetinaFace(2020) : https://docs.google.com/presentation/d/1-AR-jTFyR5w2BIIRXJfD22CdfvDTPA86/edit?usp=sharing&ouid=107012802823432471585&rtpof=true&sd=true
- `박준혁` : 비전공자로서 데이터사이언스로 작년부터 입문했습니다! 분석쪽으로 공부했었고, 엔지니어와 비전쪽은 이제 배워나가는 단계라 같이 성장했으면 좋겠어요!! (ง •̀ᴗ•́)ง 
    - https://docs.google.com/presentation/d/1G1gMfB_oJsaf5C2pGXzWdNB6gF0b_PnEQUhaWguI9Jg/edit#slide=id.geeefed5b4b_0_0
- `조혜원` : 저는 소프트웨어학과 수료생이고, 인공지능 부서에서 인턴을 하다가 저의 놀라운 감자력을 깨닫고 부스트캠프에 합류하게 되었습니다!함께 으쌰으쌰하면서 성장하고 싶어요 ٩( ᐛ )و

---
