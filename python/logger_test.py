import mgp

@mgp.read_proc
def prvi(ctx: mgp.ProcCtx) -> mgp.Record():


    logger = mgp.Logger()
    logger.info("vamos")
    logger.warning("a")
    logger.error("la")
    logger.critical("playa")

    return mgp.Record()