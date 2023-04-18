import torch
from torchvision import transforms

from baseline import init_model, ArtDataset

MODEL_WEIGHTS = "./baseline.pt"
TEST_DATASET = "./path/to/test_dataset"

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = init_model(device, num_classes=40)
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model.eval()

    img_size = 224
    trans = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    dset = ArtDataset(TEST_DATASET, transform=trans)
    batch_size = 16
    num_workers = 4
    testloader = torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    all_image_names = [item.split("/")[-1] for item in dset.files]
    all_preds = []
    model = model.eval()
    with torch.no_grad():
        for idx, (images, _) in enumerate(testloader, 0):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())

    with open("./submission.csv", "w") as f:
        f.write("image_name\tlabel_id\n")
        for name, cl_id in zip(all_image_names, all_preds):
            f.write(f"{name}\t{cl_id}\n")
