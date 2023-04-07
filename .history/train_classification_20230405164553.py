l

_epochs = 20
for epoch in range(n_epochs):
    model.train()
    for inputs, labels in train_dataloader:
        y_pred = model(inputs.to(device))
        loss = loss_fn(y_pred, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc = 0
    count = 0
    model.eval()
    acc /= count
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))