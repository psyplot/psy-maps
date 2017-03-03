import os

import subprocess as spr


def deploy(src_dir, target_branch, *what):
    p = spr.Popen('git rev-parse --verify HEAD'.split(),
                  stdout=spr.PIPE)
    p.wait()
    sha = p.stdout.read().decode('utf-8').splitlines()[0]
    work = os.getcwd()

    # change to the repository
    os.chdir(src_dir)
    p = spr.Popen('git config remote.origin.url'.split(),
                  stdout=spr.PIPE)
    repo = p.stdout.read().decode('utf-8').splitlines()[0]
    repo_name = repo.replace('https://', '')

    spr.check_call('git config user.name "Travis"'.split())
    spr.check_call(
        ('git config user.email "%s"' % (
            os.getenv('COMMIT_AUTHOR_EMAIL'))).split())

    spr.check_call('git add -N'.split() + list(what))

    if not spr.call('git diff --exit-code'.split()):
        print("No changes to the output on this push; exiting.")
        os.chdir(work)
        return

    if os.getenv('TRAVIS'):
        msg = 'Deploy from Travis job %s: Commit %s [skip ci]' % (
            os.getenv('TRAVIS_JOB_NUMBER'), sha)
        this_branch = os.getenv('TRAVIS_BRANCH')
    elif os.getenv('APPVEYOR'):
        msg = 'Deploy from Appveyor job %s: Commit %s [skip ci]' % (
            os.getenv('APPVEYOR_BUILD_VERSION'), sha)
        this_branch = os.getenv('APPVEYOR_REPO_BRANCH')
    # Commit the "changes", i.e. the new version.
    spr.check_call(('git commit -am').split() + [msg])

    # Now that we're all set up, we can push.
    # Since we push in parallel, and the remote repository might be locked, we
    # give it 10 tries
    spr.check_call(['git', 'checkout', target_branch])

    for i in range(1, 11):
        cmd = "git push https://<secure>@%s %s" % (repo_name, target_branch)
        full_cmd = "git pull && git rebase TRAVIS_DEPLOY && " + cmd
        print(('Try No. %i: ' % i) + full_cmd)

        spr.check_call('git pull --no-commit origin'.split() + [target_branch])

        spr.check_call('git rebase TRAVIS_DEPLOY'.split())
        spr.call('git commit -m'.split() + [
            msg + '\n\nMerge branch "%s" of "%s"' % (this_branch, repo)])

        p = spr.Popen(
            cmd.replace('<secure>', os.getenv('GH_REPO_TOKEN')).split(),
            stdout=spr.PIPE, stderr=spr.PIPE)
        print(p.stdout.read().decode('utf-8').replace(
                os.getenv('GH_REPO_TOKEN'), '<secure>'))
        p.wait()
        if p.poll():
            print(p.stderr.read().decode('utf-8').replace(
                os.getenv('GH_REPO_TOKEN'), '<secure>'))
            print('Failed')
            if i == 10:
                raise ValueError('Failed after 10 tries')
            print('Retrying in 10 seconds...')
        else:
            print('Success')
            break

    os.chdir(work)
