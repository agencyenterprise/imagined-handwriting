"""
Adapted from the following...
Author: Awni Hannun

This is an example CTC decoder written in Python. The code is
intended to be a simple example and is not designed to be
especially efficient.

The algorithm is a prefix beam search for a model trained
with the CTC loss function.

For more details checkout either of these references:
  https://distill.pub/2017/ctc/#inference
  https://arxiv.org/abs/1408.2873

"""
import collections
import math

from imagined_handwriting.settings import CHARACTERS

NEG_INF = -float("inf")


def make_new_beam():
    fn = lambda: (NEG_INF, NEG_INF)
    return collections.defaultdict(fn)


def logsumexp(*args):
    """
    Stable log sum exp.
    """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp


def prefix_to_str(prefix, tokenizer):
    return "".join(map(tokenizer.decode, prefix))


def decode(
    log_probs,
    beam_size=100,
    blank=0,
    trie=None,
    tokenizer=None,
    lm=None,
    lm_weight=0.2,
    space_weight=1,
):
    """
    Performs inference for the given output probabilities.
    This restricts the search to characters sequences that are in
    the trie.
    Arguments:
        probs: The output probabilities (e.g. post-softmax) for each
        time step. Should be an array of shape (time x output dim).
        beam_size (int): Size of the beam to use during inference.
        blank (int): Index of the CTC blank label.
    Returns the output label sequence and the corresponding negative
    log-likelihood estimated by the decoder.
    """
    timesteps, n_chars = log_probs.shape
    probs = log_probs

    beam = [(tuple(), (0.0, NEG_INF))]

    for t in range(timesteps):

        next_beam = make_new_beam()  # empty dictionary to hold prefix and scores

        for c in range(n_chars):
            p = probs[t, c]

            for i, (prefix, (p_b, p_nb)) in enumerate(beam):  # Loop over beam

                # If we propose a blank the prefix doesn't change.
                # Only the probability of ending in blank gets updated.
                if c == blank:
                    # if blank, the probability of ending in a blank is updated
                    #    blank prob =
                    #       current_blank_prob
                    #       + prev blank prob * decoder prob
                    #       + prev not blank prob * decoder prob
                    #
                    # note that current_blank_prob is actually always 0 since
                    # either:
                    #   1. this prefix is not in the beam
                    #   2. this prefix is in the beam because it ended in a double
                    #      character which gets collapsed to the same prefix,
                    #       in which case the current blank prob is not updated and
                    #       is still 0 - the initial value.
                    #
                    #
                    #
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_b = logsumexp(n_p_b, p_b + p, p_nb + p)
                    next_beam[prefix] = (n_p_b, n_p_nb)
                    continue

                # extend the prefix
                n_prefix = prefix + (c,)
                end_t = prefix[-1] if prefix else None
                n_p_b, n_p_nb = next_beam[n_prefix]
                if c != end_t:
                    # a new character, so we add up the probability of getting here
                    # from a blank or from not a blank
                    n_p_nb = logsumexp(n_p_nb, p_b + p, p_nb + p)
                else:
                    # The prefix contains a double letter so it had to be that
                    # we previously ended in a blank, otherwise we would have
                    # merged the double letter to a single letter.  Therefore
                    # we use the probability of getting here from a blank.
                    n_p_nb = logsumexp(n_p_nb, p_b + p)

                # *NB* this would be a good place to include an LM score.
                #
                # The probability of this new prefix.  Note that the probability
                # of ending in a blank is 0 since we just added a new character.
                # It's not until we loop around the beam again and look at this
                # extending this prefix with a blank that the probability of
                # ending in a blank is updated.

                # lets just naively score every prefix with the bigram LM
                # presumably real words will have higher scores than
                # nonsense.
                if lm is not None:
                    str_prefix = prefix_to_str(n_prefix, tokenizer)
                    if str_prefix == ">":
                        n_p_nb = NEG_INF
                    else:
                        str_prefix = str_prefix.replace(">", " ").replace("~", ".")
                        score = lm.score(
                            str_prefix, eos=False, bos=True
                        )  # -log10 score
                        if c != 27:
                            n_p_nb += lm_weight * score
                        else:
                            n_p_nb += lm_weight * score * space_weight
                        # n_p_nb += lm_weight*score
                    # print(n_prefix, n_p_b, n_p_nb)

                next_beam[n_prefix] = (n_p_b, n_p_nb)

                # If s is repeated at the end we also update the unchanged
                # prefix. This is the merging case.
                #
                # If we are ending with the same letter then we need to
                # update the probability of the prefix (without the extra
                # repeated letter)
                # to include the case where we did not end in a blank so that there
                # is no seperatation and instead we are collapsing two letters into one.
                # i.e. we need to update the probability of not ending in a blank
                # with the previous probability of not ending in a blank * the
                # probability of getting the same character this time step.
                if c == end_t:
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_nb = logsumexp(n_p_nb, p_nb + p)
                    next_beam[prefix] = (n_p_b, n_p_nb)

                # Add Trie pruning here
                # if the current prefix does not have a completion in
                # the trie, the set the probability of ending in
                # a non-blank to 0 (i.e. -inf)
                if trie is not None:
                    str_prefix = prefix_to_str(n_prefix, tokenizer)
                    word_idx = -1 if c != 27 else -2
                    word_prefix = str_prefix.split(">")[word_idx].split("'")[-1]
                    word_prefix = (
                        word_prefix.replace("~", "").replace("?", "").replace(",", "")
                    )
                    if not trie.has_subtrie(word_prefix) and not trie.has_node(
                        word_prefix
                    ):
                        next_beam[n_prefix] = (next_beam[n_prefix][0], NEG_INF)

        # Sort and trim the beam before moving on to the
        # next time-step.
        beam = sorted(next_beam.items(), key=lambda x: logsumexp(*x[1]), reverse=True)

        beam = beam[:beam_size]

    best = beam[0]
    return best[0], -logsumexp(*best[1])


class Tokenizer:
    """Simple tokenizer for converting between text and ids.

    Defaults to the characters used for imagined handwriting.

    """

    def __init__(self, characters=None):
        self.characters = characters or ["<eps>"] + CHARACTERS.abbreviated
        self.id_to_char = {i: c for i, c in enumerate(self.characters)}
        self.char_to_id = {c: i for i, c in enumerate(self.characters)}

    def decode(self, id):
        if isinstance(id, list):
            return "".join(map(self._decode, id))
        return self._decode(id)

    def _decode(self, id):
        return self.id_to_char[id]
