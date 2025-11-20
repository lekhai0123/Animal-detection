import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
import torchvision.models as models
import torchvision.transforms as T

base = "WAID"
img_base = os.path.join(base, "images")
viz_dir = os.path.join(os.path.dirname(os.path.abspath(base)), "viz")
os.makedirs(viz_dir, exist_ok=True)

splits = ["train", "valid", "test"]

def load_images(split):
    paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        paths += glob(os.path.join(img_base, split, ext))
    return paths

train_imgs = load_images("train")
val_imgs = load_images("valid")
test_imgs = load_images("test")
all_imgs = train_imgs + val_imgs + test_imgs

def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, name), dpi=300)
    plt.close()

def show_bar_split():
    c = [len(train_imgs), len(val_imgs), len(test_imgs)]
    plt.figure(figsize=(6,4))
    plt.bar(["train","val","test"], c)
    plt.title("split count")
    save_fig("split_counts.png")

def get_resolutions(paths):
    r = []
    for p in paths:
        try:
            im = Image.open(p)
            r.append(im.size)
        except:
            pass
    return r

res = get_resolutions(all_imgs)

def res_plots():
    ws = [w for w,h in res]
    hs = [h for w,h in res]
    plt.figure(figsize=(6,4))
    plt.hist(ws,bins=40)
    plt.title("width")
    save_fig("width.png")

    plt.figure(figsize=(6,4))
    plt.hist(hs,bins=40)
    plt.title("height")
    save_fig("height.png")

def aspect():
    ars=[w/h for w,h in res]
    plt.figure(figsize=(6,4))
    plt.hist(ars,bins=40)
    plt.title("aspect ratio")
    save_fig("aspect_ratio.png")

def brightness(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return np.mean(g)

def contrast(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return np.std(g)

def br_ct_plot(paths):
    br=[]
    ct=[]
    for p in paths[:2500]:
        im=cv2.imread(p)
        if im is None: continue
        br.append(brightness(im))
        ct.append(contrast(im))
    plt.figure(figsize=(6,4))
    plt.hist(br,bins=40)
    plt.title("brightness")
    save_fig("brightness.png")

    plt.figure(figsize=(6,4))
    plt.hist(ct,bins=40)
    plt.title("contrast")
    save_fig("contrast.png")

def gallery(paths,n=20):
    s=np.random.choice(paths,min(n,len(paths)),replace=False)
    cols=5
    rows=int(np.ceil(len(s)/cols))
    plt.figure(figsize=(16,10))
    for i,p in enumerate(s):
        im=cv2.imread(p)
        if im is None: continue
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        plt.subplot(rows,cols,i+1)
        plt.imshow(im)
        plt.axis("off")
    plt.suptitle("gallery")
    save_fig("gallery.png")

def file_size(paths):
    sz=[]
    for p in paths:
        try: sz.append(os.path.getsize(p)/1024)
        except: pass
    plt.figure(figsize=(6,4))
    plt.hist(sz,bins=40)
    plt.title("file size KB")
    save_fig("file_size.png")

def sharp(paths):
    vals=[]
    for p in paths[:2500]:
        im=cv2.imread(p)
        if im is None: continue
        g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        vals.append(cv2.Laplacian(g,cv2.CV_64F).var())
    plt.figure(figsize=(6,4))
    plt.hist(vals,bins=40)
    plt.title("sharpness")
    save_fig("sharpness.png")

def fft_mag(paths):
    vals=[]
    for p in paths[:500]:
        im=cv2.imread(p)
        if im is None: continue
        g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        f=np.fft.fft2(g)
        fs=np.fft.fftshift(f)
        m=20*np.log(np.abs(fs)+1e-8)
        vals.append(np.mean(m))
    plt.figure(figsize=(6,4))
    plt.hist(vals,bins=40)
    plt.title("fft spectrum")
    save_fig("fft_spectrum.png")

def rgb_hist(paths):
    r=[]
    g=[]
    b=[]
    for p in paths[:2000]:
        im=cv2.imread(p)
        if im is None: continue
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        r.append(np.mean(im[:,:,0]))
        g.append(np.mean(im[:,:,1]))
        b.append(np.mean(im[:,:,2]))
    plt.figure(figsize=(6,4))
    plt.hist(r,bins=40,color='r')
    plt.title("R")
    save_fig("R_mean.png")
    plt.figure(figsize=(6,4))
    plt.hist(g,bins=40,color='g')
    plt.title("G")
    save_fig("G_mean.png")
    plt.figure(figsize=(6,4))
    plt.hist(b,bins=40,color='b')
    plt.title("B")
    save_fig("B_mean.png")

def exposure(paths):
    vals=[]
    for p in paths[:2000]:
        im=cv2.imread(p)
        if im is None: continue
        hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        vals.append(np.mean(hsv[:,:,2]))
    plt.figure(figsize=(6,4))
    plt.hist(vals,bins=40)
    plt.title("exposure (V)")
    save_fig("exposure.png")

def noise(paths):
    vals=[]
    for p in paths[:2000]:
        im=cv2.imread(p)
        if im is None: continue
        g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        vals.append(np.mean(np.abs(g.astype(np.float32)-cv2.GaussianBlur(g,(7,7),0))))
    plt.figure(figsize=(6,4))
    plt.hist(vals,bins=40)
    plt.title("noise")
    save_fig("noise.png")

def entropy(paths):
    vals=[]
    for p in paths[:2000]:
        im=cv2.imread(p)
        if im is None: continue
        g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        hist,_=np.histogram(g, bins=256, range=(0,256))
        prob=hist/np.sum(hist)
        e=-np.sum([p*np.log2(p) for p in prob if p>0])
        vals.append(e)
    plt.figure(figsize=(6,4))
    plt.hist(vals,bins=40)
    plt.title("entropy")
    save_fig("entropy.png")

def edge_density(paths):
    vals=[]
    for p in paths[:2000]:
        im=cv2.imread(p)
        if im is None: continue
        g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        e=cv2.Canny(g,100,200)
        vals.append(np.mean(e)/255)
    plt.figure(figsize=(6,4))
    plt.hist(vals,bins=40)
    plt.title("edge density")
    save_fig("edge_density.png")

def colorfulness(paths):
    vals=[]
    for p in paths[:2000]:
        im=cv2.imread(p)
        if im is None: continue
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        rg=np.abs(im[:,:,0]-im[:,:,1])
        yb=np.abs(0.5*(im[:,:,0]+im[:,:,1])-im[:,:,2])
        cf=np.mean(np.sqrt(rg**2+yb**2))
        vals.append(cf)
    plt.figure(figsize=(6,4))
    plt.hist(vals,bins=40)
    plt.title("colorfulness")
    save_fig("colorfulness.png")

device="cpu"
model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc=torch.nn.Identity()
model.eval()
model.to(device)
trans=T.Compose([
    T.Resize((224,224)),
    T.ToTensor()
])

def get_features(paths,n=300):
    feats=[]
    used=[]
    for p in paths[:n]:
        im=Image.open(p).convert("RGB")
        x=trans(im).unsqueeze(0).to(device)
        with torch.no_grad():
            f=model(x).cpu().numpy().flatten()
        feats.append(f)
        used.append(p)
    return np.array(feats), used

feats, used = get_features(all_imgs, 400)

def pca_plot():
    p=PCA(n_components=2)
    x=p.fit_transform(feats)
    plt.figure(figsize=(6,5))
    plt.scatter(x[:,0],x[:,1],s=10)
    plt.title("PCA images")
    save_fig("pca.png")

def tsne_plot():
    t=TSNE(n_components=2,perplexity=30,learning_rate=200)
    x=t.fit_transform(feats)
    plt.figure(figsize=(6,5))
    plt.scatter(x[:,0],x[:,1],s=10)
    plt.title("t-SNE images")
    save_fig("tsne.png")

def kmeans_plot(k=5):
    km=KMeans(n_clusters=k)
    cl=km.fit_predict(feats)
    plt.figure(figsize=(6,5))
    plt.scatter(feats[:,0],feats[:,1],c=cl,s=10)
    plt.title("KMeans feature clusters")
    save_fig("kmeans.png")

def similarity_matrix():
    m=np.dot(feats,feats.T)
    m=m/np.max(m)
    plt.figure(figsize=(7,6))
    plt.imshow(m,cmap='viridis')
    plt.title("similarity matrix")
    save_fig("similarity_matrix.png")

def brightness_heat():
    vals=[]
    for p in used:
        im=cv2.imread(p)
        if im is None: vals.append(0)
        else: vals.append(brightness(im))
    arr=np.array(vals).reshape(int(np.sqrt(len(vals))),-1)
    plt.figure(figsize=(6,5))
    plt.imshow(arr,cmap='inferno')
    plt.title("brightness heatmap")
    save_fig("brightness_heatmap.png")

def freq_heat():
    vals=[]
    for p in used:
        im=cv2.imread(p)
        if im is None: vals.append(0)
        else:
            g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            f=np.fft.fft2(g)
            fs=np.fft.fftshift(f)
            vals.append(np.mean(np.abs(fs)))
    arr=np.array(vals).reshape(int(np.sqrt(len(vals))),-1)
    plt.figure(figsize=(6,5))
    plt.imshow(arr,cmap='magma')
    plt.title("frequency heatmap")
    save_fig("frequency_heatmap.png")

def texture_energy(paths):
    vals=[]
    for p in paths[:2000]:
        im=cv2.imread(p)
        if im is None: continue
        g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        sobx=cv2.Sobel(g,cv2.CV_64F,1,0)
        soby=cv2.Sobel(g,cv2.CV_64F,0,1)
        vals.append(np.mean(np.sqrt(sobx**2+soby**2)))
    plt.figure(figsize=(6,4))
    plt.hist(vals,bins=40)
    plt.title("texture energy")
    save_fig("texture_energy.png")

show_bar_split()
res_plots()
aspect()
br_ct_plot(all_imgs)
gallery(train_imgs)
file_size(all_imgs)
sharp(all_imgs)
fft_mag(all_imgs)
rgb_hist(all_imgs)
exposure(all_imgs)
noise(all_imgs)
entropy(all_imgs)
edge_density(all_imgs)
colorfulness(all_imgs)
pca_plot()
tsne_plot()
kmeans_plot()
similarity_matrix()
brightness_heat()
freq_heat()
texture_energy(all_imgs)
