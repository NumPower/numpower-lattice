<?php
namespace NumPower\Lattice\Models;

use \NDArray as nd;
use NumPower\Lattice\Layers\ILayer;
use NumPower\Lattice\Layers\Input;
use NumPower\Lattice\Optimizers\IOptimizer;

class Stack extends Model
{
    /**
     * @param ILayer $layer
     * @return Model
     */
    public function add(ILayer $layer): Model
    {
        $this->layers[] = $layer;
        return $this;
    }

    /**
     * @return Model
     */
    public function pop(): Model
    {
        $this->layers = array_pop($this->layers);
        return $this;
    }

    public function save(): void
    {

    }

    public function fit(\NDArray $X, \NDArray $y, int $epochs = 10, bool $use_gpu = False): void
    {
        for ($current_epoch = 0; $current_epoch < $epochs; $current_epoch++) {
            $this->getEpochPrinter()->start($current_epoch, $epochs);
            $loss = parent::trainStep([$X, $y]);
            $this->getEpochPrinter()->stop();
            print("loss: $loss\n");
        }
    }

    public function predict(\NDArray $x): array
    {
        $predictions = [];
        foreach ($x as $sample) {
            $outputs = $this->forward(nd::reshape($sample, [1, count($sample)]));
            $predictions[] = $outputs[0];
        }
        return $predictions;
    }
}