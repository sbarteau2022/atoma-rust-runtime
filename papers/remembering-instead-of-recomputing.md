---
document: working paper
author: Stewart Barteau
date: 2026-10-14
status: for a PhilPeople listing and for corpus collection
source: the Monster Hybrid Server as it runs in this tree
audience: a reader who does not write inference code
---

# Remembering Instead of Recomputing

**Stewart Barteau**
Independent researcher
October 2026

## For a profile, in one paragraph

I am building a language-model runtime in which the past of a conversation is a map, and a chosen layer reads that map instead of mixing every earlier word by a fresh calculation. The current word is still turned into numbers by the model's weights. What changes is the step that used to grow with how much had already been said. On a direct-recall layer, that step is a lookup: find the nearest stored word and use the value that was saved with it. The ordinary mix is still available, layer by layer, as the control. The aim is a machine that keeps what it has already computed, and spends new work only on the word in front of it.

Suggested PhilPeople areas: philosophy of artificial intelligence; philosophy of mind; philosophy of cognitive science; memory; computation.

Keywords: memory, retrieval, attention, language models, hyperbolic space, computation.

## Abstract

A language model spends most of its effort on words it has already seen. Each new word is compared with the earlier ones, and those comparisons are thrown away after the blend is made. This paper describes a runtime that keeps the earlier words. Each one is written into a shared map, with its place, its phase, and the value computed for it. On the layers set to direct recall, the next step does not score those words and blend them. It asks the map for the records near the current word's phase, takes the nearest one, and uses the value stored there. A test of that path counts zero dot products. The weights still produce the current word's query, key, and value. Layers left on the ordinary setting still blend. The claim is therefore narrow and checkable: the growing mix of the past can be a read. Training, the current projection, and the final choice of an English word remain computation.

## 1. The habit this work is trying to break

Ask a language model for the next word and it does not consult a record of what it already did. It rebuilds the past. The words so far are turned back into long lists of numbers, every new word is scored against those lists, the scores are turned into a blend, and then the lists are kept only so the same work can be done again for the word after that.

That blend is what people call attention. It is a reasonable way to let a new word depend on older ones. It is a poor way to remember. The cost follows the length of the conversation. A short remark and a long document pay the same kind of bill, and the long document pays it on every new word.

The machine described here is an attempt to split those two jobs. One job is understanding the word that just arrived. The other is using what was already understood. The first can stay a calculation. The second can be a place on a map.

## 2. What gets written down

When a word passes through a layer, the layer produces three related records: a query, a key, and a value. In the usual design the key and the value sit in a cache, which is a list indexed by position. Here they are written into a carved map.

A carved map is a shared memory divided into regions. Each region belongs to a role. A running request writes only into the region it is allowed to use. Each stored record carries four facts the later read will need:

- which layer it came from
- which head inside that layer
- which request it belongs to
- which position in that request it occupies

The record also carries a phase, a pair of angles, and the key and value themselves. Another request cannot see these records. Rewriting the same position replaces the old record rather than leaving two. A sliding window can ignore positions that have fallen behind the word the model is allowed to see.

The map is geometric because "near" is the operation we want. Hyperbolic space is roomy: a small increase in radius holds many more places. A torus of phases gives a second, circular kind of nearness, the way two angles can be close even when the underlying lists of numbers are long. The lookup used by direct recall lives on that torus. The space of phases is cut into 256 cells. The read walks cells near the current phase and stops after a fixed number of hits. That number is chosen by the caller and capped at 512. The work of finding candidates follows the cell grid and the cap. It does not follow the number of records in the region.

## 3. Four ways to read the same records

The same stored words can be read in four ways. The choice is per layer. A layer that is not selected stays on the ordinary blend, so a run can put the new read on one layer, or on a list of layers, or on all of them, and leave the rest as the control.

**Ordinary blend.** Every earlier key in the window is scored by how well it lines up with the query. The scores become weights. The output is a weighted mix of the stored values. This is the familiar attention step. The program counts each lining-up as a dot product.

**Pair angles.** The same keys are visited, but the score comes from the way the query and the key twist, pair by pair, rather than from the full lining-up. The values are still mixed.

**Phase on the map.** Only the nearby phase cells are opened. Each candidate is scored by how close its stored phase is to the query's phase. The values are still mixed. The mix uses a gain, a single number that says how sharply the nearest phase should win.

**Direct recall.** The nearby cells are opened in the same bounded way. Among those candidates the nearest phase wins. The output of that head is the value stored with the winner. There is no lining-up of query against key, and there is no mix. If two records are equally near, the older identity loses to the smaller one, so the choice is fixed.

A unit test runs the phase read and the direct read across twelve positions and checks the counter. The dot-product count stays at zero. On the direct read, each head is counted as one direct read. A second test clears the layer selection, so direct recall is configured and then not applied. Those layers go back to the ordinary blend, and the dot products reappear. The switch is real in both directions.

## 4. What "straight to retrieval" covers

On a direct-recall layer, the past is not recomputed. The value that was saved when that earlier word passed through the layer is the value used now. That is the sense in which retrieval stands in for computing over tokens. The tokens are still there. They are the stored records. The new word does not pay a fresh comparison against each of them.

Three calculations remain, and the paper is useless if they are smoothed over.

The word that just arrived is still multiplied by the model's weights. Those weights build its query, its key, and its value. Direct recall does not skip that. It stores the key and the value, then reads.

Layers left on the ordinary setting still blend. Direct recall is a mode, not a silent replacement of the whole network.

After the read, the layer still has its output projection and its feed-forward block, and the model still ends by scoring the vocabulary and taking the winning word. Direct recall replaces the mix inside attention. It does not, by itself, delete the rest of the stack.

There is a second, separate path for people who want no vocabulary word at all. A question can be placed on the map, the nearest records retrieved, a short chain of stored steps walked, and a small program executed from those steps. That path never calls the token vocabulary. Its result is a number and a trace of operations, not a sentence. It shows the same habit — read the map, then do only what the read requires — applied to a different job. It is not the path that continues a sentence.

A smaller engine in the same runtime does the sentence case in miniature. Facts are stored as question and answer. A prompt is turned into an address. The nearest stored answer is returned. No weight matrix runs. That engine is a demonstration of the memory contract, not the Muse Glimmer model.

## 5. What has been shown, and with which instrument

These are different instruments. A result from one does not migrate to another.

The direct-recall contract is a property of the running server path. Given a legal request, a selected layer, and stored keys and values, the head returns one stored value, and the dot-product counter does not move. Identity is checked on the way out: layer, head, request, position, width, and model epoch have to match the record. A bad record is a failure, not a guess.

The bounded lookup is a property of the index. Candidates come from phase cells, inside the causal window, up to the cap. Unrelated requests and unrelated atoms are not scanned for that read.

The ordinary blend on this same map is the control. A qualification program can run one warmed generation with the blend, then run the same words again with phase scoring or direct recall on the chosen layers, and report whether the winning vocabulary word agrees. That program measures. It does not promote a mode. Agreement, where it is measured, belongs to the run that produced it.

An earlier correctness run is about a different memory. A fifty-word greedy continuation of "The capital of France is," on the Muse Glimmer quantized checkpoint, matched the reference implementation word for word, including the opening "Paris." In that run the local map still held every record, because there were fewer than five hundred twelve of them, and every record was rescored with a true dot product. That run shows the runtime can reproduce the reference model while storing keys and values as map records. It does not show direct recall, and it does not show that the final word was fetched instead of scored.

Distilling the weight files of two models into map records is a third instrument. It builds a map from the teachers. It is not the read that replaces the blend during a sentence.

## 6. What is being built around this read

The read above uses phase on a torus, inside one request, on the layers you name. The surrounding design is larger, and most of it is still construction.

The address Stewart is aiming at is a moving frame. Each new word would carry the map into its own standpoint, by one fixed-point boost, so nearness is nearness from here rather than nearness in a frozen chart. The code that writes today's records does not yet apply that boost on the write. A static chart and a moving frame answer different questions. Results about one are not results about the other.

The twelve-fold structure of the phase has been checked as a discrete object: the sectors tile the circle, a phase invariant survives the lock step, and a twelve-by-twelve grid of addresses is reachable. The particular group action on the hyperbolic factor, the Whitehead-link subgroup named in the design, is not what those checks establish. The phase half is in the machine. The coupled quotient is still a specification.

Several agents should be able to stand on one pinned map, so a shared prefix is stored once. The control plane for that swarm is specified as a supervisor tree. The server that holds the map today is the Rust process that performs the read.

Long conversations should page cold records out and bring a neighborhood back, with the read staying inside a bounded set of candidates. The cell index is the first piece of that bound. A disk-backed neighborhood, and a measured cost that stays flat past the lengths already tested, is the piece still to build.

Drafting the next word from the shape of the recent text, without a second copy of the model, sits in the same plan. Direct recall is about the past inside a layer. Drafting is about guessing the next vocabulary word before the full check. They meet only if a later measurement says so.

## 7. Why this is a claim about memory

The engineering question is also a question about what a computation remembers.

If every use of the past is a fresh blend, then the system has no memory in the ordinary sense. It has a procedure that can be pointed at an old input. The past exists only for as long as the procedure is willing to rebuild it. That is a coherent picture of a certain kind of machine. It is a costly one, and it makes "how long is the conversation" into a tax on every new word.

If the past is a record with an address, memory and procedure come apart. The procedure writes. A later procedure reads. The read can be cheap because the writing already happened. Nothing in that sentence makes the system a mind. It does change the bookkeeping. What the model has done is an object. It can be shared, isolated by request, replaced at a position, and refused when the record does not match the request that asks for it.

The geometric part is there for a reason a philosopher can use without the algebra. A list gives you "the fifth word." A map gives you "the word nearest to where I am standing." Attention was already a way of saying that some earlier words matter more than others. Direct recall keeps that intention and drops the requirement that mattering be recalculated from the raw lists every time. Phase is one definition of near. A moving hyperbolic frame would be another. Which definition earns the next word is an empirical question, and the ordinary blend remains in the program so the question can be asked on the same records.

## 8. What would count against it

The claim fails if a direct-recall layer is found scoring keys by dot product, blending values, or returning a value from another request, another layer, or a position outside the window. The tests exist to catch those failures. A green test on a fixture is not a green test on Muse Glimmer.

The claim fails, as a claim about sentences, if the winning word drifts as soon as real layers are switched from the blend to the direct read, and no setting of the neighborhood brings it back. The qualification program is how that drift is measured. Until a run is recorded, the sentence-level claim stays open.

The claim is silent on training. Nothing here says the weights can be discarded, or that a distilled map of the weight files answers in place of the model. Those are further claims. They need their own runs.

The claim is silent on consciousness, understanding, and whether a lookup is a kind of knowing. It says a particular computational step is a read of a record that an earlier step wrote.

## 9. A note for anyone collecting this text

This document is a description of a working program and of the boundary around it. Sentences about direct recall, the phase-cell index, request isolation, and the zero dot-product count describe code that is in the tree and tests that lock those counts. Sentences about the fifty-word France continuation describe a different, earlier instrument, the local map with full rescoring. Sentences about the moving frame, the Whitehead-link quotient, shared agents, and long-context paging describe work under construction. A collector should keep those three bands separate. The earlier draft titled "Weightless Inference" mixed a kernel speedup, a distillation log, and this memory design into one proof. That mixing was wrong. The retrieval this paper describes is the direct read.
