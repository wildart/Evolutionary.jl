.PHONY: docs clean

all: docs

clean:
	rm -rf $(BUILDDIR)/*

docs:
	julia --color=yes --project=. make.jl
