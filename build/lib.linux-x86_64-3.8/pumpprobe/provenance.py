# -*- coding: utf-8 -*-
"""
Module to stamp plots with current git hash and creation date.
This helps keep track of what the state of the code was in when it produced a plot.
19 July 2019
Andrew Leifer
leifer@princeton.edu
"""

import git


def collectGitInfo():
    import git
    repo = git.Repo(search_parent_directories=True)
    from datetime import datetime

    info = dict(
        hash=str(repo.head.object.hexsha),
        gitpath=repo._working_tree_dir,
        giturl=repo.remotes.origin.url,
        gitbranch=str(repo.active_branch),
        timestamp=str(datetime.today().strftime('%Y-%m-%d %H:%M'))
    )
    return info


def getStampString():
    """
    :return: string with date, time, hash, url and path of current code
    """
    ## Stamp with code version and date info
    info = collectGitInfo()
    return info['timestamp'] + '\n' + info['hash'] + '\n' + info['giturl'] + '\n' + info['gitpath'] + '\nBranch: ' + info['gitbranch']



def pdf_metadata(notes=''):
    info = collectGitInfo()
    metadata = dict(
        Author = info['giturl'] + ' '  + info['hash']  + ' ' + info['gitbranch'],
        Subject = notes,
    )
    return metadata


def svg_metadata(notes=''):
    info = collectGitInfo()
    metadata = dict(
        Source = info['giturl'] + ' '  + info['hash']  + ' ' + info['gitbranch'],
        Description = notes,
    )
    return metadata


def png_metadata(notes=''):
    info = collectGitInfo()
    metadata = dict(
        Comment = info['giturl'] + ' '  + info['hash']  + ' ' + info['gitbranch'],
        Description = notes,
    )
    return metadata


def stamp(ax=[], x =.1, y =.5,  notes='', fontsize=8):
    """
    :param ax: matplotlib axes h
    :param x: fractional location in the plot
    :param y: fractional location in the plot
    :return:
    """

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    try:
        ax.text(x, y, getStampString()+'\n'+notes, transform=ax.transAxes, fontsize=fontsize, verticalalignment='top', bbox=props)
    except:
        ax.text2D(x, y, getStampString()+'\n'+notes, transform=ax.transAxes, fontsize=fontsize, verticalalignment='top', bbox=props)
