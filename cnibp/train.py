from tqdm import tqdm


def train(model, dataset, loss_list, optimizer, scheduler, epoch):
    model.train()
    cost_sum = 0
    dbp_cost_sum = 0
    sbp_cost_sum = 0
    scale_cost_sum = 0
    total_cost_sum = 0

    with tqdm(dataset, desc='Train-{}'.format(str(epoch)), total=len(dataset),
              leave=True) as train_epoch:
        for idx, (ppg, _, abp, _, d, s, info) in enumerate(train_epoch):
            optimizer.zero_grad()
            pred_abp, dbp, sbp, amp = model(ppg)
            cost = loss_list[0](pred_abp, abp)
            # amp_cost = loss_list[-2](dbp, sbp, amp, d, s, amp)
            # dbp_cost, sbp_cost, scale_cost = loss_list[-1](dbp, sbp, amp, d, s, amp)
            # amp_cost_sum = loss_list[-1](amp, d, s)

            # amp_cost = loss_list[-1](dbp, sbp, mbp, d, s, m)
            # total_cost = cost + dbp_cost + sbp_cost + scale_cost
            # total_cost = cost + dbp_cost + sbp_cost + scale_cost# + amp_cost_sum

            cost_sum += cost.item()
            avg_cost = cost_sum / (idx + 1)
            # dbp_cost_sum += dbp_cost.item()
            # dbp_avg_cost = dbp_cost_sum / (idx + 1)
            # sbp_cost_sum += sbp_cost.item()
            # sbp_avg_cost = sbp_cost_sum / (idx + 1)
            # scale_cost_sum += scale_cost.item()
            # scale_avg_cost = scale_cost_sum / (idx + 1)
            # amp_cost_sum += amp_cost.item()
            # amp_avg_cost = amp_cost_sum / (idx + 1)
            # total_cost_sum += total_cost.item()
            total_avg_cost = total_cost_sum / (idx + 1)

            postfix_dict = {}

            postfix_dict['corr'] = round(avg_cost, 3)
            # postfix_dict['amp'] = round(amp_avg_cost, 3)
            # postfix_dict['dbp'] = round(dbp_avg_cost, 3)
            # postfix_dict['sbp'] = round(sbp_avg_cost, 3)
            # postfix_dict['scale'] = round(scale_avg_cost, 3)
            # postfix_dict['total'] = round(total_avg_cost, 3)

            train_epoch.set_postfix(losses=postfix_dict)
            cost.backward()
            optimizer.step()
        scheduler.step()

    return total_avg_cost
