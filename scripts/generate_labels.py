#!/usr/bin/env python3

# %%
from github import Auth, Github
import os
import seaborn as sns


# %%
def get_repo():
    token = os.getenv("GITHUB_TOKEN")  # or do your favorite auth thing here
    auth = Auth.Token(token)
    g = Github(auth=auth)
    g.get_user().login
    repo = g.get_repo("digital-earths-global-hackathon/hk25-teams")
    return repo


def get_contents(repo):
    """
    Get the contents of the README.md files in the groups directory.
    """
    # Get the contents of the README.md files in the groups directory
    # and return a dictionary with the group name as the key and the
    # title of the README.md file as the value.
    root_contents = repo.get_contents("")
    groups = [
        g for g in root_contents if g.type == "dir" and g.name.startswith("hk25-")
    ]
    contents = dict()
    for x in groups:
        print(x)
        contents[x.name] = repo.get_contents(
            f"{x.name}/README.md"
        ).decoded_content.splitlines()[0]
    contents = {
        k: v.decode("utf-8").split("(")[0].strip()
        if "(" in v.decode("utf-8")
        else v.decode("utf-8")
        for k, v in contents.items()
    }
    contents = {k: v.replace("# ", "").strip() for k, v in contents.items()}
    return contents


def create_labels(repo, contents):
    labels = repo.get_labels()
    label_names = set(x.name for x in labels)
    cp = sns.color_palette("husl", len(contents)).as_hex()
    i = 0
    for k, v in contents.items():
        if k not in label_names:
            repo.create_label(k, cp[i][1:], v)
        i += 1


# %%
repo = get_repo()
contents = get_contents(repo)
# %%

create_labels(repo, contents)
