import torch
from torch.autograd import Variable
import setting as st
import train_model

num_epochs = 100
num_batchsize = 100
learning_rate = 0.0005

train_path = 'train/train_label.csv'

model = train_model.CNN(num_classes=st.MAX_CAPTCHA*st.ALL_CHAR_SET_LEN).cuda()
model.train()
# model.load_state_dict(torch.load('./model/CNN_14_layers_epoch20_loss0.001109_acc_99.79757085020243.pkl'))
print('Init ## {} ## model.'.format(model.model_name()))

criterion = torch.nn.MultiLabelSoftMarginLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the Model
train_dataloader = st.train_data_loader(num_batchsize, train_path)

print(model)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        images = Variable(images).cuda()
        labels = Variable(labels.float()).cuda()
        predict_labels = model(images).cuda()

        loss = criterion(predict_labels, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch:", epoch, "loss:", loss.item())
    if (epoch) % 5 == 0 and epoch != 0:
        numCor = st.test_data(model)
        model.train()
        pklname = "./model/{0}_epoch{1}_loss{2}_acc_{3}.pkl".format(
            model.model_name(), epoch, str(loss.item())[:8], numCor)
        torch.save(model.state_dict(), pklname)
        print("save model_"+pklname)

torch.save(model.state_dict(), "./model/{0}.pkl".format(model.model_name()))
print("save last model")
