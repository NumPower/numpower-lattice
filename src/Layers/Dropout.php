<?php

namespace NumPower\Lattice\Layers;

use \NDArray as nd;
use NumPower\Lattice\Core\Layers\Layer;
use NumPower\Lattice\Core\Variable;
use NumPower\Lattice\Exceptions\ValueErrorException;

class Dropout extends Layer
{
    /**
     * @var float
     */
    private float $rate;

    /**
     * @param float $rate
     * @throws ValueErrorException
     */
    public function __construct(float $rate)
    {
        $this->setDropoutRate($rate);
        parent::__construct('dropout_'.substr(uniqid(), -4), false);
    }

    /**
     * @param float $rate
     * @return void
     * @throws ValueErrorException
     */
    public function setDropoutRate(float $rate): void
    {
        if ($rate < 0 || $rate >= 1) {
            throw new ValueErrorException('Argument $rate must be a scalar in the range [0, 1)');
        }
        $this->rate = $rate;
    }

    /**
     * @param array $inputShape
     * @return void
     */
    public function build(array $inputShape): void
    {
        $this->setBuilt(true);
        $this->setInputShape($inputShape);
    }

    /**
     * @param Variable $inputs
     * @param bool $training
     * @return Variable
     */
    public function __invoke(Variable $inputs, bool $training = True): Variable
    {
        if (!$this->built()) {
            $this->build($inputs->getShape());
        }
        if ($training) {
            $mask = nd::random_binominal($inputs->getShape(), 1, 1 - $this->rate);
        } else {
            return $inputs;
        }
        return Variable::multiply($inputs, Variable::divide(Variable::fromArray($mask), (1 - $this->rate)));
    }

    /**
     * @return array
     */
    public function generateOutputShape(): array
    {
        return $this->getInputShape();
    }
}