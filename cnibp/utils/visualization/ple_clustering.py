
def get3dplot(dataset):
    train_dataset = dataset[0]
    plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.2)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(0, 6)
    ax.set_ylim3d(220, 0)
    ax.set_zlim3d(0, 4)
    ax.set_xlabel('time (t)')
    ax.set_ylabel('ABP (mmHg)')
    ax.set_zlabel('PLE')
    t_real = np.arange(0, 6, 1 / 60)
    t = np.zeros(360)
    # cmin, cmax = 0, 2
    cnt = 0
    low_cnt = 0
    high_cnt = 0
    # total_cnt = low_cnt + high_cnt
    for ple, abp, d, s, size in train_dataset:
        cnt += 1
        # color = np.array([(cmax - cmin) * np.random.random_sample() + cmin for i in range(360)])
        ple_sig = ple.view(360, ).detach().cpu().numpy()
        abp_sig = abp.view(360, ).detach().cpu().numpy()
        if np.mean(ple_sig) < 1.:
            low_cnt += 1
            ax.plot(t_real, abp_sig, zs=0, zdir='z', c='k', alpha=0.5, linewidth=0.5)
            ax.plot(t_real, zs=0, ys=ple_sig, zdir='y', c='k', alpha=0.5, linewidth=0.5)
            ax.scatter(t, abp_sig, ple_sig, c='b', marker='.', alpha=random.randint(25, 50) * 0.02, s=2)
            ax.scatter(t_real, abp_sig, ple_sig, c='b', marker='.', alpha=random.randint(25, 50) * 0.02, s=3)
        else:
            if high_cnt > 0:
                continue
            else:
                high_cnt += 1
                ax.plot(t_real, abp_sig, zs=0, zdir='z', c='k', alpha=0.5, linewidth=0.5)
                ax.plot(t_real, zs=0, ys=ple_sig, zdir='y', c='k', alpha=0.5, linewidth=0.5)
                ax.scatter(t, abp_sig, ple_sig, c='r', marker='.', alpha=random.randint(25, 50) * 0.02, s=2)
                ax.scatter(t_real, abp_sig, ple_sig, c='r', marker='.', alpha=random.randint(25, 50) * 0.02, s=3)

        if low_cnt + high_cnt > 1 and low_cnt > 0:
            break
    # ax.view_init(elev=20., azim=-35)
    plt.show()
