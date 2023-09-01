<?php

namespace NumPower\Lattice\Optimizers;

use Exception;
use NDArray as nd;
use NumPower\Lattice\Core\Models\IModel;
use NumPower\Lattice\Core\Optimizers\Optimizer;
use NumPower\Lattice\Core\Variable;
use NumPower\Lattice\Models\Model;

class Adam extends Optimizer
{
    /**
     * @var float
     */
    private float $learningRate;

    /**
     * @var float
     */
    private float $beta1;

    /**
     * @var float
     */
    private float $beta2;

    /**
     * @var float
     */
    private float $epsilon;

    /**
     * @var int
     */
    private int $t;

    /**
     * @var nd[]
     */
    private array $m;

    /**
     * @var nd[]
     */
    private array $v;

    public function __construct(float $lr = 0.001, float $beta1 = 0.9, float $beta2 = 0.999, float $epsilon = 1e-7)
    {
        $this->learningRate = $lr;
        $this->beta1 = $beta1;
        $this->beta2 = $beta2;
        $this->epsilon = $epsilon;
    }

    /**
     * @param IModel $model
     * @return void
     */
    public function build(IModel $model): void
    {
        $this->m = [];
        $this->v = [];
        foreach ($model->getLayers() as $idx => $layer) {
            if (!$layer->isTrainable()) {
                continue;
            }
            foreach ($layer->getTrainableWeights() as $w_idx => $w) {
                $this->m[$idx][$w_idx] = nd::zeros($w->getArray()->shape());
                $this->v[$idx][$w_idx] = nd::zeros($w->getArray()->shape());
            }
        }
        $this->t = 0;
    }

    /**
     * @param Variable $error
     * @param Model $model
     * @return void
     * @throws Exception
     */
    public function __invoke(Variable $error, Model $model): void
    {
        $error->backward();
        $this->t += 1;
        foreach ($model->getLayers() as $idx => $layer) {
            if (!$layer->isTrainable()) {
                continue;
            }
            foreach ($layer->getTrainableWeights() as $w_idx => $w) {
                $grad = $w->diff();
                $this->m[$idx][$w_idx] = $this->beta1 * $this->m[$idx][$w_idx] + (1 - $this->beta1) * $grad;
                $this->v[$idx][$w_idx] = $this->beta2 * $this->v[$idx][$w_idx] + (1 - $this->beta2) * $grad ** 2;
                $m_hat = $this->m[$idx][$w_idx] / (1 - $this->beta1 ** $this->t);
                $v_hat = $this->v[$idx][$w_idx] / (1 - $this->beta2 ** $this->t);
                $w->overwriteArray($w->getArray() - (($this->learningRate * $m_hat / (nd::sqrt($v_hat) + $this->epsilon))));
            }
        }
    }
}
