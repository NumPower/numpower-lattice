<?php
namespace NumPower\Lattice\Models;

use \NDArray as nd;
use NumPower\Lattice\Layers\ILayer;
use NumPower\Lattice\Layers\Input;
use NumPower\Lattice\Losses\ILoss;
use NumPower\Lattice\Optimizers\IOptimizer;

abstract class Model implements IModel
{
    private IOptimizer $optimizer;

    /**
     * @var ILayer[]
     */
    protected array $layers = [];
    private ILoss $lossFunction;

    public function setOptimizer(IOptimizer $optimizer): void
    {
        $this->optimizer = $optimizer;
    }

    /**
     * @return array
     */
    public function layers(): array
    {
        return $this->layers;
    }

    /**
     * @param nd $target
     * @param nd $output
     * @return nd|float
     */
    public function computeLoss(\NDArray $target, \NDArray $output): \NDArray|float
    {
        return $this->getLossFunction()->calculate($target, $output);
    }

    /**
     * @param \NDArray $inputs
     * @return \NDArray
     */
    public function forward(\NDArray $inputs): \NDArray
    {
        foreach ($this->layers() as $idx => $layer) {
            $inputs = $layer->forward($inputs);
        }
        return $inputs;
    }

    /**
     * Back propagation
     *
     * @param nd $error
     * @return void
     */
    public function backward(\NDArray $error): void
    {
        $reverse_layers = array_reverse($this->layers, true);
        foreach ($reverse_layers as $idx => $layer) {
            if ($idx == count($this->layers()) - 1) {
                continue;
            }
            $error = $layer->backward($reverse_layers[$idx + 1], $error);
        }
    }

    /**
     * @return IOptimizer
     */
    public function getOptimizer(): IOptimizer
    {
        return $this->optimizer;
    }

    /**
     * @return void
     */
    public function optimize(): void
    {
        foreach ($this->layers() as $idx => $layer) {
            if (is_a($layer, Input::class)) {
                continue;
            }
            if ($layer->isOutput()) {
                continue;
            }
            $this->getOptimizer()->adjust($this->layers()[$idx - 1]->getDerivative(), $layer);
        }
    }

    /**
     * @param array $data
     * @return float
     */
    public function trainStep(array $data): float
    {
        $sum_loss = 0;
        $x = $data[0];
        $y = $data[1];
        foreach($x as $idx => $sample) {
            // Forward pass
            $outputs = $this->forward(nd::reshape($sample, [1, count($sample)]));
            $error = $y[$idx] - $outputs[0];

            // Backward pass
            $this->backward($error);
            $sum_loss += $this->computeLoss($y[$idx], $outputs);
            $this->optimize();
        }
        return $sum_loss / count($x);
    }

    /**
     * @param ILoss $loss
     * @return void
     */
    public function setLossFunction(ILoss $loss): void
    {
        $this->lossFunction = $loss; 
    }

    /**
     * @return ILoss
     */
    public function getLossFunction(): ILoss
    {
        return $this->lossFunction;
    }

    /**
     * @param IOptimizer $optimizer
     * @param ILoss $loss
     * @return void
     */
    public function build(IOptimizer $optimizer, ILoss $loss): void
    {
        $this->setLossFunction($loss);
        $this->setOptimizer($optimizer);
        $layers = $this->layers();
        $input_shape = [];

        foreach ($layers as $idx => $layer) {
            if (is_a($layer, Input::class)) {
                continue;
            }
            if (!array_key_exists($idx-1, $layers)) {
                break;
            }
            $layer->initialize($layers[$idx-1], ($idx == count($this->layers) - 1));
        }
    }
}