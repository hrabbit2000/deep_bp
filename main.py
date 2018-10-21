import deep_bp as dbp

def main():
# display some lines
    bp = dbp.DeepBP([3, 4, 2])
    bp.init_parameters()
    bp.forward_cal([1.2, 1.34, 4.21])
    bp.cal_grad([1.0, 1.0])


if __name__ == "__main__": main()
