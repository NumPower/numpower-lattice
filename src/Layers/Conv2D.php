<?php

namespace NumPower\Lattice\Layers;

use \NDArray as nd;
use NumPower\Lattice\Core\Activations\IActivation;
use NumPower\Lattice\Core\IGrad;
use NumPower\Lattice\Core\Layers\InputSpec;
use NumPower\Lattice\Core\Layers\Layer;
use NumPower\Lattice\Core\Operation;
use NumPower\Lattice\Core\Tensor;
use NumPower\Lattice\Exceptions\ValueErrorException;

class Conv2D extends Layer implements IGrad
{
    /**
     * @var int
     */
    private int $filters;

    /**
     * @var string
     */
    private string $borderMode;

    /**
     * @var string
     */
    private string $format;

    /**
     * @var int[]
     */
    private int|array $kernelSize;

    /**
     * @var array|int[]
     */
    private array $strides;

    /**
     * @var bool
     */
    private bool $useBias;

    /**
     * @var
     */
    private $rank;

    /**
     * @var Tensor
     */
    private Tensor $bias;

    /**
     * @var Tensor
     */
    private Tensor $kernel;

    /**
     * @var false|mixed
     */
    private mixed $inputChannel;

    /**
     * @var IActivation|null
     */
    private ?IActivation $activation;

    /**
     * @param int $filters
     * @param int[] $kernelSize
     * @param array $strides
     * @param string $borderMode
     * @param string $format
     * @param bool $useBias
     * @param IActivation $activation
     * @throws ValueErrorException
     */
    public function __construct(
        int          $filters,
        array        $kernelSize,
        array        $strides = [1, 1],
        string       $borderMode = "valid",
        string       $format = "channel_first",
        bool         $useBias = true,
        ?IActivation $activation = null
    )
    {
        if (!in_array($borderMode, ["valid", "same"])) {
            throw new ValueErrorException(
                "Invalid borderMode argument for Conv2D layer. Valid values: `valid`, `same`"
            );
        }
        if (!in_array($format, ["channel_last", "channel_first"])) {
            throw new ValueErrorException(
                "Invalid format argument for Conv2D layer. Valid values: `channel_last`,`channel_first`"
            );
        }
        $this->filters = $filters;
        $this->borderMode = $borderMode;
        $this->format = $format;
        $this->kernelSize = $kernelSize;
        $this->strides = $strides;
        $this->useBias = $useBias;
        $this->rank = 2;
        $this->activation = $activation;
        parent::__construct("conv2d_" . substr(uniqid(), -4), true);
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

        $rows = $this->calculateOutputLength($this->kernelSize[0], $this->borderMode, $this->strides[0], $rows);
        $cols = $this->calculateOutputLength($this->kernelSize[1], $this->borderMode, $this->strides[1], $cols);
        if ($this->format == "channel_first") {
            // channel_last
            return [$this->getInputShape()[0], $this->filters, $rows, $cols];
        }
        // channel first
        return [$this->getInputShape()[0], $rows, $cols, $this->filters];
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
     * @param array $inputShape
     * @return void
     * @throws ValueErrorException
     */
    public function build(array $inputShape): void
    {
        if ($this->format == 'channel_last') {
            $inputChannel = end($inputShape);
        } else {
            $inputChannel = $inputShape[1];
        }
        $kernel_shape = array_merge($this->kernelSize, [$inputChannel, $this->filters]);
        $this->inputChannel = $inputChannel;
        $this->kernel = $this->addWeight(
            name: "kernel",
            shape: $kernel_shape,
            trainable: true
        );
        if ($this->useBias) {
            $this->bias = $this->addWeight(
                name: "bias",
                shape: [$this->filters],
                trainable: true
            );
        }
        $inputSpec = new InputSpec(shape: $inputShape, minNdim: $this->rank + 2);
        $this->setInputSpec($inputSpec);
        parent::build($inputShape); // TODO: Change the autogenerated stub
    }

    /**
     * @param Tensor $inputs
     * @param bool $training
     * @return Tensor
     * @throws ValueErrorException
     */
    public function __invoke(Tensor $inputs, bool $training = false): Tensor
    {
        if (!$this->built()) {
            $this->build($inputs->getShape());
        }
        $outputshape = $this->generateOutputShape();
        $outputshape[0] = $this->getBatchSize();
        $output = nd::zeros($outputshape);

        if ($this->format == "channel_first") {
            for ($batch_index = 0; $batch_index < $inputs->getShape()[0]; $batch_index++) {
                for ($filter_index = 0; $filter_index < $this->filters; $filter_index++) {
                    for ($channel_index = 0; $channel_index < $inputs->getArray()[$batch_index]->shape()[0]; $channel_index++){
                        $output[$batch_index][$filter_index] = nd::correlate2d($inputs->getArray()[$batch_index][$channel_index],
                            nd::transpose($this->kernel->getArray())[$filter_index][$channel_index],
                            mode: "valid",
                            boundary: "fill");
                    }
                }
            }
        } else {
            throw new ValueErrorException("`channel_last` format not implemented.");
        }

        $output = Tensor::fromArray($output);
        $output->setInputs([$inputs, $this->kernel]);
        $output->registerOperation("convolve2d", $this);
        if ($this->activation) {
            $output = ($this->activation)($output);
        }
        if ($this->useBias) {
            $output = Tensor::add($output, $this->bias);
        }
        return $output;
    }

    /**
     * @param float|int|nd $grad
     * @param Operation $op
     * @return void
     * @throws \Exception
     */
    public function backward(float|int|nd $grad, Operation $op): void
    {
        $outputshape = $this->getInputSpec()->getShape();
        $outputshape[0] = $this->getBatchSize();
        $d_input = nd::zeros($outputshape);
        $d_filters = nd::zeros($this->getTrainableWeights()[0]->shape());
        for ($f_index = 0; $f_index < $this->filters; $f_index++) {
            for ($input_idx = 0; $input_idx < $this->getInputShape()[0]; $input_idx++) {
                for ($c_index = 0; $c_index < $this->inputChannel; $c_index++) {
                    $d_filters[$f_index][$c_index] = nd::correlate2d($op->getArgs()[0]->getArray()[$input_idx][$c_index], $grad[$input_idx][$f_index], mode: "valid", boundary: "fill");
                    $d_input[$input_idx][$c_index] = nd::correlate2d($grad[$input_idx][$f_index], $op->getArgs()[1]->getArray()[$input_idx][$c_index], mode: "full", boundary: "fill");
                }
            }
        }
        $op->getArgs()[0]->backward($d_input);
        $op->getArgs()[1]->backward($d_filters);
    }
}
