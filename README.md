# Data analysis
- Document here the project: PinkFloydLyrics
- Description: Project Description
- Data Source: https://www.kaggle.com/datasets/joaorobson/pink-floyd-lyrics?resource=download
- Type of analysis: NLP with LSTM and Word embedding

# The goal of the project is to analyse every Pink Floyd lyric, and then build a model that can return a new song based on user input of 2 key words.

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for PinkFloydLyrics in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/PinkFloydLyrics`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "PinkFloydLyrics"
git remote add origin git@github.com:{group}/PinkFloydLyrics.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
PinkFloydLyrics-run
```

# Install

Go to `https://github.com/{group}/PinkFloydLyrics` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/PinkFloydLyrics.git
cd PinkFloydLyrics
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
PinkFloydLyrics-run
```
