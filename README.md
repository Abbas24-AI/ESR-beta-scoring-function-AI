# clone
git clone https://github.com/<your-org>/ESR_beta_New.git
cd ESR_beta_New

# create & activate
conda create -n erbeta python=3.10 -y
conda activate erbeta

# core Python dependencies
pip install -r requirements.txt
