<?php

namespace NumPower\Lattice\Layers;

use \NDArray as nd;
use NumPower\Lattice\Core\IGrad;
use NumPower\Lattice\Core\Layers\InputSpec;
use NumPower\Lattice\Core\Layers\Layer;
use NumPower\Lattice\Core\Operation;
use NumPower\Lattice\Core\Tensor;
use NumPower\Lattice\Exceptions\ValueErrorException;

class MaxPooling2D extends Layer implements IGrad
{
    /**
     * @var int[]
     */
    private array $poolSize;

    /**
     * @var string
     */
    private string $borderMode;

    /**
     * @var string
     */
    private string $format;

    /**
     * @var int[]|null
     */
    private ?array $strides;

    /**
     * @param int[] $poolSize
     * @param int[]|null $strides
     * @param string $borderMode
     * @param string $format
     */
    public function __construct(array $poolSize, ?array $strides = null, string $borderMode = 'valid', string $format = "channel_first")
    {
        $this->poolSize = $poolSize;
        $this->borderMode = $borderMode;
        $this->format = $format;
        $this->strides = $strides;
        parent::__construct("pooling2d_" . substr(uniqid(), -4), true);
    }

    /**
     * @param Tensor $inputs
     * @param bool $training
     * @return Tensor
     */
    public function __invoke(Tensor $inputs, bool $training = false): Tensor
    {
        $shape = $this->generateOutputShape();
        $shape[0] = count($inputs->getArray());
        $output = nd::zeros($shape);

        [$batch_size, $height, $width, $filters] = $inputs->getShape();

        for ($index = 0; $index < $batch_size; $index++) {
            foreach ($this->iterateRegions($inputs->getArray()[$index]) as $region) {
                $max = nd::max(nd::max($region[0], axis: 1), axis: 0);
                $output[$index][$region[1]][$region[2]] = $max;
            }
        }
        $tensor = Tensor::fromArray($output);
        $tensor->setInputs([$inputs]);
        $tensor->registerOperation("maxpooling_2d", $this);
        return $tensor;
    }

    /**
     * @param array $inputShape
     * @return void
     * @throws ValueErrorException
     */
    public function build(array $inputShape): void
    {
        $this->setInputSpec(new InputSpec(shape: $inputShape));
        if (!isset($this->strides)) {
            $this->strides = $this->poolSize;
        }
        parent::build($inputShape);
    }

    /**
     * @param nd $target
     * @return array
     */
    private function iterateRegions(nd $target): iterable
    {
        [$height, $width] = $target->shape();
        $new_height = intdiv($height, $this->poolSize[0]);
        $new_width = intdiv($width, $this->poolSize[1]);

        for ($i = 0; $i < $new_height; $i++) {
            for ($j = 0; $j < $new_width; $j++) {
                $region = $target->slice([$i * $this->poolSize[0], $i * $this->poolSize[0] + $this->poolSize[0]], [$j * $this->poolSize[1], $j * $this->poolSize[1] + $this->poolSize[1]]);
                yield [$region, $i, $j];
            }
        }
    }

    /**
     * @return array
     */
    public function generateOutputShape(): array
    {
        if ($this->format == "channel_first") {
            $rows = $this->getInputShape()[2];
            $cols = $this->getInputShape()[3];
        } else {
            $rows = $this->getInputShape()[1];
            $cols = $this->getInputShape()[2];
        }

        $rows = $this->calculateOutputLength($this->poolSize[0], $this->borderMode, $this->strides[0], $rows);
        $cols = $this->calculateOutputLength($this->poolSize[1], $this->borderMode, $this->strides[1], $cols);

        if ($this->format == "channel_first") {
            // channel_first
            return [$this->getInputShape()[0], $this->getInputShape()[1], $rows, $cols];
        }
        // channel_last
        return [$this->getInputShape()[0], $rows, $cols, $this->getInputShape()[3]];
    }

    /**
     * @param int $filterSize
     * @param string $borderMode
     * @param int $stride
     * @param int|null $inputLength
     * @return int|null
     */
    private function calculateOutputLength(
        int    $filterSize,
        string $borderMode,
        int    $stride,
        ?int   $inputLength = null
    ): ?int
    {
        if (!isset($inputLength)) {
            return null;
        }
        assert(in_array($borderMode, ["valid", "same"]));
        if ($borderMode == 'same') {
            $output_length = $inputLength;
        }
        if ($borderMode == 'valid') {
            $output_length = $inputLength - $filterSize + 1;
        }
        return intdiv(($output_length + $stride - 1), $stride);
    }

    /**
     * @param float|int|nd $grad
     * @param Operation $op
     * @return void
     * @throws \Exception
     */
    public function backward(float|int|nd $grad, Operation $op): void
    {
        $d_input = nd::zeros($op->getArgs()[0]->getShape());

        [$batch_size, $height, $width, $filters] = $this->getInputShape();
        $batch_size = $this->getBatchSize();

        for ($index = 0; $index < $batch_size; $index++) {
            foreach ($this->iterateRegions($op->getArgs()[0]->getArray()[$index]) as $region) {
                [$height, $width, $filter] = $region[0]->shape();
                $amax = nd::max(nd::max($region[0], axis: 1), axis: 0);

                for ($i = 0; $i < $height; $i++) {
                    for ($j = 0; $j < $width; $j++) {
                        for ($f = 0; $f < $filter; $f++) {
                            if ($region[0][$i][$j][$f] == $amax[$f]) {
                                $d_input[$index][$i * 2 + $i][$j * 2 + $j][$f] = $grad[$index][$i][$j][$f];
                            }
                        }
                    }
                }
            }
        }
        $op->getArgs()[0]->backward($d_input);
    }
}
