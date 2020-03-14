def print_progress(start, done, total, progress_bar_len=18):
    '''
        print animated progress 'frame'

        label: label for what's in progress
        done: amount of items completed
        total: total number of items
        progress_bar_len: length of progress bar to print
    '''
    progress = int(done / total * progress_bar_len)
    print_len = len(str(total)) * 2 + 1

    print(  \
            start + '\t' + \
            '[' + ('=' * progress) + '>' + (' ' * (progress_bar_len - progress)) + ']\t' + \
            ('{0:>' + str(print_len) + '}').format(str(done) + '/' +str(total)), end='\r' \
         )

def print_progress_done(start, total, progress_bar_len=18, end=''):
    '''
        print completed progress bar

        label: label for what's in progress
        total: total number of items
        progress_bar_len: length of progress bar to print
        end: optional ending message that replaces the final 'total/total'
    '''
    print_len = len(str(total)) * 2 + 1
    if(end):
        print(  \
                start + '\t' + \
                '[' + ('=' * (progress_bar_len + 1)) + ']\t' + \
                end \
             )
    else:
        print(  \
                start + '\t' + \
                '[' + ('=' * (progress_bar_len + 1)) + ']\t' + \
                ('{0:>' + str(print_len) + '}').format(str(total) + '/' +str(total)) \
             )
