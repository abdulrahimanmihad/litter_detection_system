# Deploying the Litter Detection app to AWS EC2 (HTTPS + free domain)

Goal: a clickable, secure link for your CV, e.g. **https://mihad-litter.duckdns.org**,
serving `src/app.py` (Streamlit) on an EC2 instance behind nginx with a Let's Encrypt
certificate.

Architecture:

```
Browser ──HTTPS(443)──► nginx ──proxy──► Streamlit (127.0.0.1:8501) ──► YOLOv8 (best.pt)
```

Symbols used below:
- 💻 = run on **your Windows machine** (PowerShell or Git Bash)
- 🖥️ = run on the **EC2 server** (after you SSH in)

Replace these placeholders throughout:
- `mihad-litter.duckdns.org` → your DuckDNS domain
- `<ELASTIC_IP>` → the Elastic IP you allocate in Step 2
- `litter-key.pem` → your EC2 key file

---

## Step 0 — Push your latest code to GitHub 💻

Your local fixes to `app.py` and the new `deploy/` folder aren't on GitHub yet, and
`best.pt` is git-ignored (you'll copy it up separately in Step 6).

```bash
git add src/app.py deploy/ CLAUDE.md
git commit -m "Fix app import, add EC2 deploy config"
git push origin main
```

---

## Step 1 — Launch the EC2 instance (AWS Console)

- **AMI:** Ubuntu Server 24.04 LTS (64-bit x86)
- **Instance type:** `t3.small` (2 GB RAM). Do **not** use `t2.micro`/`t3.micro` (1 GB) —
  PyTorch + YOLO inference will run out of memory. `t3.small` is ~$15/month; stop the
  instance when you don't need it to avoid charges (see Step 10).
- **Key pair:** create/download one (e.g. `litter-key.pem`). Keep it safe.
- **Storage:** 20 GB gp3 (default 8 GB is too small for torch).
- **Security group** — add inbound rules:
  | Type  | Port | Source          | Why                        |
  |-------|------|-----------------|----------------------------|
  | SSH   | 22   | My IP           | your admin access          |
  | HTTP  | 80   | Anywhere 0.0.0.0/0 | certbot + redirect to HTTPS |
  | HTTPS | 443  | Anywhere 0.0.0.0/0 | the actual public site     |

  You do **not** open 8501 — nginx reaches Streamlit internally.

---

## Step 2 — Give it a stable IP (Elastic IP)

An instance's public IP changes every stop/start, which would break your CV link.
Attach a fixed one:

AWS Console → **EC2 → Elastic IPs → Allocate** → then **Actions → Associate** it with
your instance. Note this address as `<ELASTIC_IP>`.

(An Elastic IP is free while associated with a running instance.)

---

## Step 3 — Point a free domain at it (DuckDNS)

1. Go to https://www.duckdns.org and sign in (Google/GitHub).
2. Create a subdomain, e.g. `mihad-litter`.
3. Set its **current ip** to `<ELASTIC_IP>` and click **update ip**.
4. Confirm it resolves (💻): `nslookup mihad-litter.duckdns.org` → should show `<ELASTIC_IP>`.

---

## Step 4 — Connect and prep the server 🖥️

SSH in (💻 to start the session):

```bash
ssh -i litter-key.pem ubuntu@<ELASTIC_IP>
```

Now on the server 🖥️ — add swap (cheap insurance against OOM during pip install / inference):

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

Install system packages (nginx, python venv, and the libs OpenCV needs on a headless box):

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-venv python3-pip nginx libgl1 libglib2.0-0 git
```

---

## Step 5 — Get the code 🖥️

```bash
cd ~
git clone https://github.com/abdulrahimanmihad/litter_detection_system.git
cd litter_detection_system
```

---

## Step 6 — Copy the model file up 💻

`best.pt` is git-ignored, so `git clone` didn't include it. Copy it from your machine.
Open a **new** terminal on your Windows machine (💻), in the repo folder:

```bash
scp -i litter-key.pem src/best.pt ubuntu@<ELASTIC_IP>:/home/ubuntu/litter_detection_system/src/best.pt
```

Verify on the server 🖥️: `ls -lh ~/litter_detection_system/src/best.pt` (~50 MB).

---

## Step 7 — Python environment 🖥️

```bash
cd ~/litter_detection_system
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# CPU-only torch first — avoids downloading the multi-GB CUDA build on a CPU instance
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r deploy/requirements-serve.txt
```

Quick smoke test (🖥️, Ctrl+C after it says the model loaded / server started):

```bash
streamlit run src/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

---

## Step 8 — Run it as a service (auto-start, auto-restart) 🖥️

```bash
sudo cp deploy/litter-app.service /etc/systemd/system/litter-app.service
sudo systemctl daemon-reload
sudo systemctl enable --now litter-app
sudo systemctl status litter-app        # should say "active (running)"
```

If it's not running, check logs: `journalctl -u litter-app -n 50 --no-pager`.

---

## Step 9 — nginx + HTTPS 🖥️

Put the reverse-proxy config in place (edit the domain first):

```bash
sudo cp deploy/nginx-litter.conf /etc/nginx/sites-available/litter
sudo sed -i 's/mihad-litter.duckdns.org/YOUR-DOMAIN.duckdns.org/' /etc/nginx/sites-available/litter
sudo ln -s /etc/nginx/sites-available/litter /etc/nginx/sites-enabled/litter
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

At this point `http://YOUR-DOMAIN.duckdns.org` should already show the app.

Now add the free HTTPS certificate:

```bash
sudo snap install --classic certbot
sudo ln -s /snap/bin/certbot /usr/bin/certbot
sudo certbot --nginx -d YOUR-DOMAIN.duckdns.org
```

Answer the prompts (email, agree to terms, choose redirect HTTP→HTTPS). certbot edits
the nginx config for you and sets up auto-renewal.

---

## Step 10 — Done ✅

Your CV link: **https://YOUR-DOMAIN.duckdns.org**

Put it on your CV as a hyperlink on text like *"Live Demo"* or the project title.

### Keeping costs down
- **Stop** the instance from the AWS console when not showing it — you're billed for
  compute only while running. The Elastic IP + DuckDNS domain persist, so the link keeps
  working after you start it again. (Note: an Elastic IP that's allocated but *not* on a
  running instance incurs a small charge, so leave it associated.)
- If you want it always-on cheaply, `t3.small` is the practical floor for this model.

### Updating the app later
```bash
cd ~/litter_detection_system && git pull
sudo systemctl restart litter-app
```

### Troubleshooting
- **502 Bad Gateway** → Streamlit isn't running: `sudo systemctl status litter-app` and `journalctl -u litter-app -n 50`.
- **App loads but detection errors** → check `best.pt` copied correctly (Step 6).
- **certbot fails** → make sure DuckDNS points to `<ELASTIC_IP>` and port 80 is open to the world in the security group.
- **Killed / crashes during inference** → out of memory; confirm swap is on (`free -h`) and you're on `t3.small`, not micro.
