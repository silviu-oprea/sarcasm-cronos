import enchant
import itertools
import os
import re
import string
import sys
import unicodedata

from unidecode import unidecode
import nltk
# /Applications/Python 3.6/Install Certificates.command
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import TweetTokenizer

# ====================================================================================== #
# Definitions
positive_emojis = {
    ':p', ':-p'
          ':-]', ':]', ':-3', ':3', ':->', ':>',  # smileys
    '8-)', '8)', ':-}', ':}', ':o)', ':c)',  # smileys
    ':-))))', ':-)))', ':))))', ':)))', ':))', ':-))', ':)', ';)', ':-)',
    ':‑)', ':)', ':^)', '=]', '=)',  # smileys
    '☺', '🙂', '😊', '😀', '😁',  # smileys
    ":'‑)", ":')", '😂',  # tears of laughter
    ':*', ':-*', ':x', '😗', '😙', '😚', '😘', '😍',  # kissing
    'O:‑)', 'O:)', '0:‑3', '0:3', '0:‑)',
    '0:)', '0;^)', '😇', '👼',  # angel, saint, innocent
    '#‑)',  # partied all night
    '<3'  # love
}

negative_emojis = {
    ':-(', ':(', ':‑c', ':c', ':‑<', ':<',  # frown, sad, angry, pouting
    ':‑[', ':[', ':-||', '>:[', ':{', ':@', '>:(',
    '☹', '🙁', '😠', '😡', '😞', '😟', '😣', '😖',
    ":'‑(", ":'(", '😢', '😭',  # crying
    # horror, disgust, saddness, great dismay
    "D‑':", "D:<", "D:", "D8", "D;", "D=", "DX",
    '😨', '😧', '😦', '😱', '😫', '😩',
    # skeptical, annoyed, undecided, uneasy, hesitant
    ':‑/', ':/', ':‑.', '>:\\', '>:/',
    ':\\', '=/', '=\\', ':L', '=L', ':S',
    '🤔', '😕', '😟',
    '>:‑)', '>:)', '}:‑)', '}:)', '3:‑)', '3:)', '>;)',  # evil
    '😈',
    ':‑###..', ':###..',  # being sick
    '🤒', '😷',
    "',:-|", "',:-l",  # scepticism, disbelief, or disapproval
    '<:‑|',  # dumb, dunce-like
    '%‑)', '%)',  # drunk, confused
    '😵', '😕', '🤕'
}


emojis = positive_emojis | negative_emojis
escaped_positive_emojis = set(re.escape(w) for w in positive_emojis)
escaped_negative_emojis = set(re.escape(w) for w in negative_emojis)
escaped_emojis = escaped_positive_emojis | escaped_negative_emojis

punctuation = {'"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':',
               ';', '<', '=', '>', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~',
               '?', '!', "''", '``'}
ascii_letters = set(string.ascii_lowercase + string.ascii_uppercase)
number_re = re.compile(
    '[-+]?' # optional sign
    '(?:'
    '(?:\d*\.\d+)' # .1 .12 .123 etc 9.1 etc 98.1 etc
    '|'
    '(?:\d+\.?)' # 1. 12. 123. etc 1 12 123 etc
    ')'
    # followed by optional exponent part if desired
    '(?:[Ee][+-]?\d+)?'
)
ellipsis_re = re.compile('\.\.+')

# ====================================================================================== #
unicode_chars = (chr(i) for i in range(sys.maxunicode))
control_unicode_chars = ''.join(c for c in unicode_chars
                                if unicodedata.category(c) == 'Cc')
control_char_re = re.compile('[%s]' % re.escape(control_unicode_chars))

def remove_control_chars(s):
    return control_char_re.sub('', s)

# ====================================================================================== #
# Delongation
us_dict = enchant.Dict("en_US")
uk_dict = enchant.Dict("en_GB")
au_dict = enchant.Dict("en_AU")
twitter_dict = enchant.request_pwl_dict(
    os.path.join(os.path.dirname(__file__), 'assets', 'twitter_jargon.txt'))


def is_known_word(word):
    return us_dict.check(word) \
           or uk_dict.check(word) \
           or au_dict.check(word) \
           or twitter_dict.check(word)

delongate_pattern = re.compile(r"(.)\1{2,}")
def delongate(text):
    return delongate_pattern.sub(r"\1\1", text)

def all_consecutive_duplicates_edits(word, max_repeat=float('inf')):
    chars = [[c*i for i in range(min(len(list(dups)), max_repeat), 0, -1)]
             for c, dups in itertools.groupby(word)]
    return map(''.join, itertools.product(*chars))

def delongate_to_valid_word(word):
    # An initial reduction to at most 3 repetitions.
    # This reduces processing time dramatically.
    if is_known_word(word):
        return word
    word = delongate(word)
    variants = itertools.islice(all_consecutive_duplicates_edits(word), 100)
    try:
        return next((e for e in variants if e and is_known_word(e)))
    except StopIteration:
        return word

def unrepeat(words, mark_repetition=True):
    unrepeated_words = []
    current_idx = 0
    while current_idx < len(words):
        running_idx = current_idx + 1
        while running_idx < len(words) and words[current_idx] == words[running_idx]:
            running_idx += 1
        if running_idx > current_idx + 1 and mark_repetition:
            unrepeated_words.append(markers['repeat'])
        unrepeated_words.append(words[current_idx])
        current_idx = running_idx
    return unrepeated_words


# ============================================================================ #
# Word tokenization
def word_tokenize(text):
    text = remove_control_chars(text)
    tokens = nltk_tweet_tokenizer.tokenize(text)
    tokens2 = []
    for token in tokens:
        if is_special(token):
            tokens2.append(token)
        else:
            for token2 in nltk.word_tokenize(token):
                tokens2.append(token2)
    return tokens2


# ============================================================================ #
# Tweet tokenization
nltk_tweet_tokenizer = TweetTokenizer()
nltk_word_tokenizer = nltk.word_tokenize
markers = {
    'user': '<user>',
    'hashtag': '<hashtag>',
    'url': '<url>',
    'number': '<number>',
    'elong': '<elong>',
    'emoji': '<emoji>',
    'pos_emoji': '<pos_emoji>',
    'neg_emoji': '<neg_emoji>',
    'repeat': '<repeat>',
    'ellipsis': '<ellipsis>'
}
def is_user(word): return word.startswith('@')
def is_hashtag(word): return word.startswith('#') and len(word) > 1 and word[1] != '#'
def is_url(word): return '://' in word
def is_number(word): return number_re.match(word.lower()) is not None
def is_positive_emoji(word): return word.lower() in positive_emojis
def is_negative_emoji(word): return word.lower() in negative_emojis
def is_emoji(word): return word.lower() in emojis
def is_ellipsis(word): return ellipsis_re.match(word) is not None
def is_special(word):
    return is_user(word) or is_hashtag(word) or is_url(word) \
           or is_number(word) or is_emoji(word) or is_ellipsis(word)

def tokenize_tweet(text,
                   allowed_punctuation=None,
                   remove_punctuation=True,
                   mark_repetition=True,
                   delongate=True,
                   tokenize_user=True,
                   tokenize_hashtag=True,
                   tokenize_url=True,
                   tokenize_emoji=True,
                   tokenize_number=True,
                   tokenize_ellipsis=True,
                   tokenize_all=True,
                   lowercase=True,
                   stopwords=None):
    if allowed_punctuation is None:
        allowed_punctuation = {}

    if stopwords is None:
        stopwords = {}
    if lowercase:
        stopwords = {w.lower() for w in stopwords}

    text = unidecode(text)
    words = word_tokenize(text)
    if mark_repetition:
        words = unrepeat(words, mark_repetition=True)
    tokenized_words = []

    for word in words:
        if lowercase:
            word = word.lower()
        if word in stopwords:
            continue
        if remove_punctuation and word not in allowed_punctuation and word in punctuation:
            continue

        if (tokenize_all or tokenize_ellipsis) and is_ellipsis(word):
            tokenized_words.append(markers['ellipsis'])
            continue

        if delongate:
            delongated_word = delongate_to_valid_word(word)
            if word != delongated_word:
                tokenized_words.append(markers['elong'])
            word = delongated_word

        if (tokenize_all or tokenize_user) and is_user(word):
            tokenized_words.append(markers['user'])
            continue

        if (tokenize_all or tokenize_hashtag) and is_hashtag(word):
            tokenized_words.append(markers['hashtag'])
            word = word[1:]

        if (tokenize_all or tokenize_url) and is_url(word):
            tokenized_words.append(markers['url'])
            continue

        if (tokenize_all or tokenize_number) and is_number(word):
            tokenized_words.append(markers['number'])
            continue

        if (tokenize_all or tokenize_emoji) and is_positive_emoji(word):
            tokenized_words.append(markers['pos_emoji'])
            continue

        if (tokenize_all or tokenize_emoji) and is_negative_emoji(word):
            tokenized_words.append(markers['neg_emoji'])
            continue

        if (tokenize_all or tokenize_emoji) and is_emoji(word):
            tokenized_words.append(markers['emoji'])
            continue

        tokenized_words.append(word)

    return tokenized_words


if __name__ == '__main__':
    sample_tweet = """
    school's didn't
        Great to see @AGPamBondi launch a cutting-edge statewide school school schooooool 
        safety APP in Florida today - nameeeed by Parkland Survivors :)) ....... ... ??? #dude 
        BIG PRIORITY and Florida is getting it (done)!!! #FortifyFL :(, :-(
        " and "wazzup dude"" "" 
    """
    allowed_punctuation={'.', '!', '?', '(', ')'}
    tokens = tokenize_tweet(sample_tweet, allowed_punctuation=allowed_punctuation)
    print(tokens)
