# Reduce a list of words into only lowercase and one encounter.
# Also strips {-,*,/} from the words, as they might be imortant 
# Inside the general string.

swords = set(['able', 'about', 'above', 'actual', 'add', 'added', 'again', 'all',
          'along', 'also', 'always', 'an', 'and', 'angry', 'any', 'anywhere',
          'apart', 'are', 'as', 'at', 'avoid', 'award-winning', 'away',
          'back', 'be', 'before', 'below', 'best', 'both', 'bring', 
          'bringing', 'but', 'buy', 'by', 'can', 'caution', 'check', 'choise', 
          'choose', 'demand', 'detail', 'do', 'does', 'done', 'down', 
          'duty', 'each', 'ease', 'easily', 'easy', 'effortlessly', 'either', 
          'enjoy', 'enjoying', 'enough', 'entire', 'even', 'ever', 'every', 
          'everything', 'find', 'follows', 'for', 'forget', 'free', 'from', 
          'full', 'fully', 'get', 'gives', 'giving', 'go', 'good', 'great', 
          'greater', 'group', 'handy', 'happen', 'has', 'have', 'help', 
          'helpful', 'helps', 'his', 'how', 'ideal', 'i' , 'if', 'in', 
          'is', 'it', 'its', 'its', 'itself', 'just', 'keeps', 'known', 
          'lets', 'like', 'local', 'lot', 'loud', 'make', 'many', 
          'manyfind', 'may', 'missing', 'more', 'more', 'most', 'much', 
          'nearly', 'need', 'never', 'new', 'newly', 'next', 'no', 'nook', 
          'not', 'now', 'of', 'off', 'offer', 'on', 'only', 'opt', 'or', 
          'other', 'out', 'over', 'page', 'pass', 'paten', 'patented', 
          'perfect', 'perfectly', 'please', 'quote', 'read', 
          'rich', 'right', 'same', 'see', 'select', 'show', 'shows', 'simple', 
          'since', 'sir', 'sleek', 'so', 'some', 'soon', 'specifically', 
          'such', 'sure', 'take', 'takes', 'takes', 'than', 'thanks', 'that', 
          'thats', 'the', 'their', 'them', 'themselves', 'then', 'there', 
          'theres', 'theres', 'these', 'they', 'this', 'those', 'to', 
          'together', 'top', 'try', 'typical', 'under', 'unlike', 'up', 'us', 
          'useful', 'user', 'using', 'want', 'was', 'we', 
          'well', 'whats', 'when', 'which', 'while', 'who', 'will', 
          'with', 'with', 'without', 'wont', 'wont', 'you', 'youll', 
          'your', 'youre', 'youve'])

def reducer(line,ext):
    NL = []
    line.sort()
    for l in line:
        l = l.strip('-')
        l = l.strip('*')
        l = l.strip('/')
        l = l.strip('.')
        l = l.strip('?')
        if len(l) <1:
            continue
        if not any(l.lower() in n for n in NL):
            NL.append(l.lower())
    if ext:
        setNL = set(NL).difference(swords)
        NL = list(setNL)
#    NL.sort()
    return(NL)
