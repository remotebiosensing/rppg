import matplotlib.pyplot as plt


def plot_pie_chart(title, labels, sizes, counterclock=False, autopct='%.1f%%', shadow=True):
    explode = [0.05, 0.10, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05, 0.05, 0.1][:len(labels)]
    colors = ['#ff9999', '#ffc000', '#8fd9b6', '#d395d0', '#ff9999', 'silver', 'gold', 'lightcyan', 'lightgrey',
              'dodgerblue', 'violet', '#e35f62', 'seagreen'][:len(labels)]
    plt.figure(figsize=(10, 10))
    plt.title('{}'.format(title))
    plt.pie(sizes, labels=labels, explode=explode, colors=colors,
            counterclock=counterclock, autopct=autopct, shadow=shadow)
    plt.show()
    plt.close()
