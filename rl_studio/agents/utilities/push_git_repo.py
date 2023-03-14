import subprocess

def git_add_commit_push(commit_message):
    # Change this to your repository directory
    repo_directory = "/home/ruben/Desktop/2020-phd-ruben-lucas"

    # Change to the repository directory
    subprocess.run(["git", "add", "."], cwd=repo_directory)

    # Commit changes with the given commit message
    subprocess.run(["git", "commit", "-m", commit_message], cwd=repo_directory)

    # Push changes to the remote repository
    subprocess.run(["git", "push"], cwd=repo_directory)

if __name__ == "__main__":
    commit_message = "Your commit message here."
    git_add_commit_push(commit_message)