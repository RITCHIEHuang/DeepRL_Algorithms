import click

from Algorithms.tf2.TRPO.trpo import TRPO


@click.command()
@click.option(
    "--env_id", type=str, default="MountainCar-v0", help="Environment Id"
)
@click.option(
    "--render", type=bool, default=True, help="Render environment or not"
)
@click.option(
    "--num_process",
    type=int,
    default=1,
    help="Number of process to run environment",
)
@click.option(
    "--lr_v", type=float, default=3e-4, help="Learning rate for Value Net"
)
@click.option("--gamma", type=float, default=0.99, help="Discount factor")
@click.option("--tau", type=float, default=0.95, help="GAE factor")
@click.option(
    "--max_kl", type=float, default=1e-2, help="kl constraint for TRPO"
)
@click.option("--damping", type=float, default=1e-2, help="damping for TRPO")
@click.option("--batch_size", type=int, default=1000, help="Batch size")
@click.option(
    "--model_path",
    type=str,
    default="trained_models",
    help="Directory to store model",
)
@click.option("--seed", type=int, default=1, help="Seed for reproducing")
@click.option("--test_epochs", type=int, default=1, help="Epochs for testing")
def main(
    env_id,
    render,
    num_process,
    lr_v,
    gamma,
    tau,
    max_kl,
    damping,
    batch_size,
    model_path,
    seed,
    test_epochs,
):
    trpo = TRPO(
        env_id,
        render,
        num_process,
        batch_size,
        lr_v,
        gamma,
        tau,
        max_kl,
        damping,
        seed=seed,
        model_path=model_path,
    )

    for i_iter in range(1, test_epochs):
        trpo.eval(i_iter)


if __name__ == "__main__":
    main()
