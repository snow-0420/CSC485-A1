#!/usr/bin/env python3
# Student name: Richard Yan
# Student number: 1005731193
# UTORid: yanrich2

import typing as T
from math import inf

import torch
import torch.nn.functional as F
from torch import Tensor


def is_projective(heads: T.Iterable[int]) -> bool:
    """
    Determines whether the dependency tree for a sentence is projective.

    Args:
        heads: The indices of the heads of the words in sentence. Since ROOT
          has no head, it is not expected to be part of the input, but the
          index values in heads are such that ROOT is assumed in the
          starting (zeroth) position. See the examples below.

    Returns:
        True if and only if the tree represented by the input is
          projective.

    Examples:
        The projective tree from the assignment handout:
        >>> is_projective([2, 5, 4, 2, 0, 7, 5, 7])
        True

        The non-projective tree from the assignment handout:
        >>> is_projective([2, 0, 2, 2, 6, 3, 6])
        False
    """
    projective = True
    # *#* BEGIN YOUR CODE *#* #
    for i in range(len(heads)):
        if not projective:
            break
        left = min(i + 1, heads[i])
        right = max(i + 1, heads[i])
        for j in range(left + 1, right):
            if heads[j - 1] < left or heads[j - 1] > right:
                projective = False
                break
    # *** END YOUR CODE *** #
    return projective


def is_single_root_tree(heads: Tensor, lengths: Tensor) -> Tensor:
    """
    Determines whether the selected arcs for a sentence constitute a tree with
    a single root word.

    Remember that index 0 indicates the ROOT node. A tree with "a single root
    word" has exactly one outgoing edge from ROOT.

    If you like, you may add helper functions to this file for this function.

    Args:
        heads (Tensor): a Tensor of dimensions (batch_sz, sent_len) and dtype
            int where the entry at index (b, i) indicates the index of the
            predicted head for vertex i for input b in the batch

        lengths (Tensor): a Tensor of dimensions (batch_sz,) and dtype int
            where each element indicates the number of words (this doesn't
            include ROOT) in the corresponding sentence.

    Returns:
        A Tensor of dtype bool and dimensions (batch_sz,) where the value
        for each element is True if and only if the corresponding arcs
        constitute a single-root-word tree as defined above

    Examples:
        Valid trees from the assignment handout:
        >>> is_single_root_tree(torch.tensor([[2, 5, 4, 2, 0, 7, 5, 7],\
                                              [2, 0, 2, 2, 6, 3, 6, 0]]),\
                                torch.tensor([8, 7]))
        tensor([True, True])

        Invalid trees (the first has a cycle; the second has multiple roots):
        >>> is_single_root_tree(torch.tensor([[2, 5, 4, 2, 0, 8, 6, 7],\
                                              [2, 0, 2, 2, 6, 3, 6, 0]]),\
                                torch.tensor([8, 8]))
        tensor([False, False])
    """
    # *#* BEGIN YOUR CODE *#* #
    tree_single_root = torch.ones_like(heads[:, 0], dtype=torch.bool, device=heads.device)
    # look through tree one by one
    for i in torch.arange(heads.size(dim=0), device=heads.device):
        from_root = -1
        valid = True

        # look through each word
        word_lst = list(torch.arange(1, lengths[i] + 1, dtype=torch.long, device=heads.device))
        valid_set = set()
        while word_lst:
            j = word_lst.pop(0)
            ancestor_lst = [j]
            parent = heads[i, j - 1]

            # if word j is from root directly
            if parent == 0:
                # no words from root recorded yet
                if from_root < 0:
                    from_root = j
                    ancestor_lst.append(parent)
                # some other words already from root
                else:
                    valid = False
                    break

            # check by traversing ancestors
            while 0 not in ancestor_lst:
                # found cycle
                if parent in ancestor_lst:
                    valid = False
                    break
                # if we have already verified that the parent word has a
                # path (of ancestor) to the root, then don't need to
                # traverse another time to waste time
                if int(parent) in valid_set:
                    break

                ancestor_lst.append(parent)
                parent = heads[i, parent.long() - 1]
                if parent == 0 and from_root < 0:
                    from_root = 0

            if not valid:
                break
            else:
                # word that doesn't break validity (no cycle and found 0 as
                # highest ancestor) implies that any ancestors of this word
                # will also not break, so don't do the whole loop again
                # on those words to waste time
                word_lst = [x for x in word_lst if x not in ancestor_lst]
                # record all words that is known to have a path to root
                valid_set.update([int(x) for x in ancestor_lst])

        tree_single_root[i] = valid
    # *** END YOUR CODE *** #
    return tree_single_root


def single_root_mst(arc_scores: Tensor, lengths: Tensor) -> Tensor:
    """
    Finds the maximum spanning tree (more technically, arborescence) for the
    given sentences such that each tree has a single root word.

    Remember that index 0 indicates the ROOT node. A tree with "a single root
    word" has exactly one outgoing edge from ROOT.

    If you like, you may add helper functions to this file for this function.

    Args:
        arc_scores (Tensor): a Tensor of dimensions (batch_sz, x, y) and dtype
            float where x=y and the entry at index (b, i, j) indicates the
            score for a candidate arc from vertex j to vertex i.

        lengths (Tensor): a Tensor of dimensions (batch_sz,) and dtype int
            where each element indicates the number of words (this doesn't
            include ROOT) in the corresponding sentence.

    Returns:
        A Tensor of dtype int and dimensions (batch_sz, x) where the value at
        index (b, i) indicates the head for vertex i according to the
        maximum spanning tree for the input graph.

    Examples:
        >>> single_root_mst(torch.tensor(\
            [[[0, 0, 0, 0],\
              [12, 0, 6, 5],\
              [4, 5, 0, 7],\
              [4, 7, 8, 0]],\
             [[0, 0, 0, 0],\
              [1.5, 0, 4, 0],\
              [2, 0.1, 0, 0],\
              [0, 0, 0, 0]],\
             [[0, 0, 0, 0],\
              [4, 0, 3, 1],\
              [6, 2, 0, 1],\
              [1, 1, 8, 0]]]),\
            torch.tensor([3, 2, 3]))
        tensor([[0, 0, 3, 1],
                [0, 2, 0, 0],
                [0, 2, 0, 2]])
    """
    # *#* BEGIN YOUR CODE *#* #
    # initialize the returned tensor and modified arc scores used for doing
    # contraction and expansion
    best_arcs = torch.zeros(arc_scores.size(dim=0), arc_scores.size(dim=1), dtype=torch.long, device=arc_scores.device)
    mod_arc_scores = torch.zeros_like(arc_scores, device=arc_scores.device)

    # compute the best arcs
    for i in torch.arange(arc_scores.size(dim=0), device=arc_scores.device):
        for j in torch.arange(1, lengths[i] + 1, device=arc_scores.device):
            best_arcs[i, j] = torch.argmax(arc_scores[i, j, :])
            mod_arc_scores[i, j, :] = arc_scores[i, j, :] - arc_scores[i, j, best_arcs[i, j].long()]

    # check if the resulting graph is single rooted MST or not
    is_valid = is_single_root_tree(best_arcs[:, 1:], lengths)
    # breakpoint()
    for i in torch.arange(is_valid.size(dim=0), device=arc_scores.device):
        # if the resulting graph is not a single rooted MST
        if not is_valid[i]:
            # multiple edge out from root
            if lengths[i] - torch.count_nonzero(best_arcs[i, 1:]) > 1:
                has_cycle = False
            # has a cycle
            else:
                has_cycle = True

            # case of solving a cycle
            if has_cycle:
                # create a similar graph with additionally the contracted node
                # and the indices of the cycles
                new_arc_score, cycle, from_head = contract(best_arcs[i, 1:], mod_arc_scores[i], lengths[i])
                column = torch.LongTensor([x for x in torch.arange(new_arc_score.size(dim=0), device=arc_scores.device) if x not in cycle]).to(arc_scores.device)
                row = torch.unsqueeze(column, 1)
                # recursively find the best arcs in the contracted graph
                head = torch.squeeze(
                    single_root_mst(torch.unsqueeze(new_arc_score[row, column], 0),
                                    torch.unsqueeze(lengths[i] - len(cycle) + 1, 0)), dim=0)
                # expand the contracted graph
                best_arcs[i, from_head[lengths[i] + 1, head[lengths[i] - len(cycle) + 1]]] = head[lengths[i] - len(cycle) + 1]

            # case of solving multiple edge coming out from root
            else:
                # indices of edges coming out from root
                index_from_root = [x for x in torch.arange(1, int(lengths[i] + 1), device=arc_scores.device)
                                   if not torch.isin(x, torch.nonzero(best_arcs[i, :lengths[i] + 1]))]
                new_arc_score = torch.zeros(len(index_from_root), arc_scores.size(dim=1), arc_scores.size(dim=2), device=arc_scores.device)
                new_length = torch.full((len(index_from_root), ), lengths[i], device=arc_scores.device)
                # initiate the same arc_scores, but only with one of the edge
                # coming out from the root as choose-able (all other become -inf)
                for j in range(len(index_from_root)):
                    new_arc_score[j] = arc_scores[i]
                    new_arc_score[j, [x for x in index_from_root if x != index_from_root[j]], 0] = -inf
                # compute the best arcs to those modified graphs
                new_best_arcs = single_root_mst(new_arc_score, new_length)
                # compute the best score of choosing one of them
                # (find the maximum score from picking different edges)
                scores = torch.zeros(len(index_from_root), device=arc_scores.device)
                for j in range(len(index_from_root)):
                    scores[j] = sum([x for x in arc_scores[i][list(range(1, lengths[i] + 1)),
                                                              new_best_arcs[j][list(range(1, lengths[i] + 1))].tolist()]])
                best_score_index = torch.argmax(scores, dim=0)
                best_arcs[i, :] = new_best_arcs[best_score_index]
    # *** END YOUR CODE *** #
    return best_arcs


def contract(head: Tensor, arc_score: Tensor, length: Tensor) -> (Tensor, list, Tensor):
    """
    Contract a cycle in the graph represented by head.
    head only contain a single graph, so size (x, ).
    arc_score is size (x, y).
    length is a single scalar.

    return the updated arc_score, indices of the cycle, and
    a same size tensor as arc_score to represent among all those edges
    we've discarded, which head do we choose from

    note: contracted node will be the last node in the graph, so the graph
    will have one more word than the original graph
    (i.e. words being contracted will not be deleted,
    but recorded in the returned indices)

    We discard edges because for example if we contract v_2 and v_3 to v_4
    then we have both edges root->v_2 and root->v_3 directing to root->v_4,
    but we can't handle this because we only use a matrix representation
    of arc_scores. So we pick the maximum of root->v_2 and root->v_3 to be
    the edge from root->v_4 (the lower score will always not going to be
    considered by the MST algorithm since we always choose the best
    incoming edge. Notice that only when we are in the cases of multiple
    edges from root, will we then cover some maximum incoming edge so some
    other edge (not maximum) will be considered. In cycle, this is not going
    to be the case.).
    """
    # to find the cycle
    cycle = []
    # look through each word
    word_lst = list(torch.arange(1, length + 1, device=arc_score.device))
    no_cycle_set = set()
    while word_lst:
        j = int(word_lst.pop(0))
        ancestor_lst = [j]
        parent = int(head[j - 1])

        # check by traversing ancestors
        while 0 not in ancestor_lst:
            # found cycle
            if parent in ancestor_lst:
                cycle = ancestor_lst[ancestor_lst.index(parent):]
                break
            # if we have already verified that the parent word has a
            # path (of ancestor) to the root, then don't need to
            # traverse another time to waste time
            if parent in no_cycle_set:
                break

            ancestor_lst.append(parent)
            parent = int(head[parent - 1])

        if not cycle:
            # word that doesn't break validity (no cycle and found 0 as
            # highest ancestor) implies that any ancestors of this word
            # will also not break, so don't do the whole loop again
            # on those words to waste time
            word_lst = [x for x in word_lst if x not in ancestor_lst]
            # record all words that is known to have a path to root
            no_cycle_set.update([x for x in ancestor_lst])
        else:
            # found the cycle
            break

    # guaranteed to have cycle detected and recorded until this point
    new_arc_score = arc_score.detach().clone().to(arc_score.device)
    from_head = torch.zeros_like(arc_score, dtype=torch.long, device=arc_score.device)
    if head.size(dim=0) != length:
        temp = torch.zeros_like(arc_score, dtype=torch.long, device=arc_score.device)
        # if the head has padded value in the end, we can just modify that
        new_arc_score[:, length + 1], temp[:, length + 1] = torch.max(arc_score[:, cycle], 1)
        new_arc_score[length + 1, :], temp[length + 1, :] = torch.max(arc_score[cycle, :], 0)
        from_head[:, length + 1] = torch.LongTensor([cycle[x] for x in temp[:, length + 1].tolist()]).to(arc_score.device)
        from_head[length + 1, :] = torch.LongTensor([cycle[x] for x in temp[length + 1, :].tolist()]).to(arc_score.device)

        return new_arc_score, cycle, from_head
    else:
        # if the head has no padded value in the end, we have to
        # concatenate the contracted vertex onto it
        # (i.e. a 4x4 arc_score representing 3 words (not considering root)
        # in example input, when need to contract, will concatenate the
        # contracted vertex into the end, so return arc_score will be 5x5)
        temp1, temp2 = torch.max(arc_score[:, cycle], 1)
        temp1 = torch.unsqueeze(temp1, dim=1)
        temp2 = torch.LongTensor([cycle[x] for x in temp2.tolist()]).to(arc_score.device)
        temp2 = torch.unsqueeze(temp2, dim=1)
        new_arc_score = torch.cat((new_arc_score, temp1), dim=1)
        from_head = torch.cat((from_head, temp2), dim=1)
        temp1, temp2 = torch.max(arc_score[cycle, :], 0)
        temp1 = torch.cat((temp1, Tensor([-inf]).to(arc_score.device)))
        temp2 = torch.LongTensor([cycle[x] for x in temp2.tolist()]).to(arc_score.device)
        temp2 = torch.cat((temp2, torch.LongTensor([0]).to(arc_score.device)))
        temp1 = torch.unsqueeze(temp1, dim=0)
        temp2 = torch.unsqueeze(temp2, dim=0)
        new_arc_score = torch.cat((new_arc_score, temp1), dim=0)
        from_head = torch.cat((from_head, temp2), dim=0)

        return new_arc_score, cycle, from_head


if __name__ == '__main__':
    import doctest
    doctest.testmod()
