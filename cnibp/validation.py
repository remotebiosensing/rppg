import torch
from tqdm import tqdm


def validation(model, dataset, loss_list, epoch, scaler=True):
    model.eval()
    cost_sum = 0
    dbp_cost_sum = 0
    sbp_cost_sum = 0
    # scale_cost_sum = 0
    total_cost_sum = 0

    with tqdm(dataset, desc='Validation-{}'.format(str(epoch)), total=len(dataset), leave=True) as valid_epoch:
        with torch.no_grad():
            for idx, (ppg, _, abp, _, d, s, info) in enumerate(valid_epoch):
                pred_abp, dbp, sbp = model(ppg)

                cost = loss_list[0](pred_abp, abp)
                dbp_cost = loss_list[1](dbp, d)
                sbp_cost = loss_list[2](sbp, s)

                total_cost = cost + dbp_cost + sbp_cost  # + scale_cost
                # total_cost = cost + dbp_cost + sbp_cost + scale_cost# + amp_cost_sum

                cost_sum += cost.item()
                avg_cost = cost_sum / (idx + 1)
                dbp_cost_sum += dbp_cost.item()
                dbp_avg_cost = dbp_cost_sum / (idx + 1)
                sbp_cost_sum += sbp_cost.item()
                sbp_avg_cost = sbp_cost_sum / (idx + 1)
                total_cost_sum += total_cost.item()
                total_avg_cost = total_cost_sum / (idx + 1)

                postfix_dict = {}

                postfix_dict['corr'] = round(avg_cost, 3)
                # postfix_dict['amp'] = round(amp_avg_cost, 3)
                postfix_dict['dbp'] = round(dbp_avg_cost, 3)
                postfix_dict['sbp'] = round(sbp_avg_cost, 3)
                # postfix_dict['scale'] = round(scale_avg_cost, 3)
                postfix_dict['total'] = round(total_avg_cost, 3)

                valid_epoch.set_postfix(losses=postfix_dict)  # , tot=total_cost.__float__())

        return total_avg_cost
