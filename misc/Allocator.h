/*     Copyright 2015-2017 Egor Yusov
 *  
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF ANY PROPRIETARY RIGHTS.
 *
 *  In no event and under no legal theory, whether in tort (including negligence), 
 *  contract, or otherwise, unless required by applicable law (such as deliberate 
 *  and grossly negligent acts) or agreed to in writing, shall any Contributor be
 *  liable for any damages, including any direct, indirect, special, incidental, 
 *  or consequential damages of any character arising as a result of this License or 
 *  out of the use or inability to use the software (including but not limited to damages 
 *  for loss of goodwill, work stoppage, computer failure or malfunction, or any and 
 *  all other commercial damages or losses), even if such Contributor has been advised 
 *  of the possibility of such damages.
 */

#pragma once
#include "engine/foundation/Noncopyable.h"
// #include "STDAllocator.h"
#include "Utils/Utils.h"
#include <memory>
#include <mutex>
#include <vector>
#include <queue>
#include <string>
#include <set>
#include <map>
#include <unordered_map>
#include <functional>

namespace feynman {


enum class SpaceAllocatorType {
    BestFit = 0,
    Stack = 1,
    StaticChunk = 2,
    DynamicChunk = 3,
    // SubAllocator = 4,
};

/// return entire block with head padding and unalignedOffset
///  
/// |<------------------result----------------->|
/// |<------------------block------------------>|
/// |                   |<---- alignedSize ---->|
/// |                   alignedOffset           |
/// unalignedOffset
///
struct UnalignedAllocation {
    size_t unalignedOffset;
    size_t size;

    static constexpr size_t npos = ~size_t{0};

    UnalignedAllocation() : unalignedOffset(0), size(0) {}
    UnalignedAllocation(size_t unalignedOffset_, size_t size_) : unalignedOffset{unalignedOffset_}, size{size_} {}
    
    
    bool operator==(const UnalignedAllocation& rhs) const noexcept {
        return unalignedOffset == rhs.unalignedOffset && size == rhs.size;
    }
    bool isValid() const { 
        return unalignedOffset != npos;
    }
    static UnalignedAllocation InvalidAllocation() { return UnalignedAllocation(npos, 0); }
};

/// return part of block without head padding and with alignedOffset
///  
/// |                   |<------- result ------>|
/// |<------------------block------------------>|
/// |                   |<---- alignedSize ---->|
/// |                 alignedOffset             |
/// unalignedOffset
///
struct Allocation {
    size_t offset;  // aligned offset
    size_t size;
    
    static constexpr size_t npos = ~size_t{0};

    Allocation() : offset(0), size(0) {}
    Allocation(size_t alignedOffset_, size_t size_) : offset{alignedOffset_}, size{size_} {}
    
    
    bool operator==(const Allocation& rhs) const noexcept {
        return offset == rhs.offset && size == rhs.size;
    }
    bool isValid() const { 
        return offset != npos;
    }
    static Allocation InvalidAllocation() { return Allocation(npos, 0); }
};

/// interface of universal space allocation
class IAllocator : public Noncopyable {
public:
    using Ptr = std::shared_ptr<IAllocator>;
    // Allocate Count descriptors
    virtual Allocation allocate(size_t count, size_t alignment) = 0;
    virtual void free(Allocation& allocation) = 0;
    virtual size_t getFreeSize() const = 0;
};


/// scenario: allocate shared srv for per mesh on hardware tier 1 who has limted srv per shader
/// 
/// This algorithm divide memory block into groups by size, allocate is to find best fit block.
/// There is only one block at initialized time.
/// This algorithm maybe lead to memory fragmentation.
/// alignment must power of two, returned address is alignedOffset
/// returned allocation should be wrapped with RAII object
/// 
///
/// The class handles free memory block management to accommodate variable-size allocation requests.
/// It keeps track of free blocks only and does not record allocation sizes. The class uses two ordered maps
/// to facilitate operations. The first map keeps blocks sorted by their offsets. The second multimap keeps blocks
/// sorted by their sizes. The elements of the two maps reference each other, which enables efficient block
/// insertion, removal and merging.
///
///   8                 32                       64                           104
///   |<---16--->|       |<-----24------>|        |<---16--->|                 |<-----32----->|
///
///
///        m_FreeBlocksBySize      m_FreeBlocksByOffset
///           size->offset            offset->size
///
///                16 ------------------>  8  ---------->  {size = 16, &m_FreeBlocksBySize[0]}
///
///                16 ------.   .-------> 32  ---------->  {size = 24, &m_FreeBlocksBySize[2]}
///                          '.'
///                24 -------' '--------> 64  ---------->  {size = 16, &m_FreeBlocksBySize[1]}
///
///                32 ------------------> 104 ---------->  {size = 32, &m_FreeBlocksBySize[3]}
///
/// new fix:
/// 1. busy map to track allocated block
/// 2. return aligned offset without head padding
class BestFitAllocator : public IAllocator {
public:
    // ------------------------- type define
    using Ptr = std::shared_ptr<BestFitAllocator>;
    using OffsetType = size_t;
    using SizeType = size_t;

private:
    struct FreeBlockInfo;

    // offset -> block
    // Type of the map that keeps memory blocks sorted by their offsets
    using TFreeBlocksByOffsetMap = std::map<OffsetType, FreeBlockInfo>;

    // size -> offsetmap.it
    // Type of the map that keeps memory blocks sorted by their sizes
    using TFreeBlocksBySizeMap = std::multimap<SizeType, TFreeBlocksByOffsetMap::iterator>;

    struct FreeBlockInfo {
        SizeType size;
        TFreeBlocksBySizeMap::iterator orderBySizeIt;
        FreeBlockInfo(OffsetType size_) : size(size_) {}
    };

public:

    explicit BestFitAllocator(SizeType totalSize, OffsetType minAlignment = 1): m_TotalSize(totalSize), m_FreeSize(totalSize), m_MinAlignment(minAlignment) {
        FEYNMAN_ASSERT(Utils::isPowerOf2(minAlignment), "minAlignment error");
        addNewBlock(0, m_TotalSize);
        resetCurrAlignment();
    }
    virtual ~BestFitAllocator() {
    }

    /// allocate block without head padding
    ///  
    /// |                   |<------- result ------>|
    /// |<----------------- block ----------------->|
    /// |                   |<---- alignedSize ---->|
    /// |                   alignedOffset           |
    /// 
    /// @return allocated allocation or invalid allocation
    virtual Allocation allocate(SizeType size, OffsetType alignment) override {
        // allocate entire block
        UnalignedAllocation unalignedAllocation = allocateUnaligned(size, alignment);
        if (!unalignedAllocation.isValid()) {
            // FEYNMAN_ASSERT(unalignedAllocation.isValid());
            return Allocation::InvalidAllocation();
        }
        OffsetType alignedOffset = Utils::alignTo(unalignedAllocation.unalignedOffset, alignment);
        SizeType alignedSize = unalignedAllocation.size - (alignedOffset - unalignedAllocation.unalignedOffset);

        // return part of block, with offset aligned, and tracking the allocation
        Allocation allocation{alignedOffset, alignedSize}; // todo: maybe origin size
        m_BusyBlocksByAlignedOffset.insert({alignedOffset, unalignedAllocation});
        return allocation;
    }
    virtual void free(Allocation& allocation) override {
        FEYNMAN_ASSERT(allocation.isValid());
        free(allocation.offset);
        allocation = Allocation{};
    }

    bool isFull() const{ return m_FreeSize == 0; };
    bool isEmpty()const{ return m_FreeSize == m_TotalSize; };
    OffsetType getMaxSize() const{return m_TotalSize;}
    OffsetType getFreeSize()const override {return m_FreeSize;}
    OffsetType getUsedSize()const{return m_TotalSize - m_FreeSize;}
    size_t getNumFreeBlocks() const { return m_FreeBlocksByOffset.size(); }
    size_t getNumBusyBlocks() const { return m_BusyBlocksByAlignedOffset.size(); }

    OffsetType getMaxFreeBlockSize() const {
        return !m_FreeBlocksBySize.empty() ? m_FreeBlocksBySize.rbegin()->first : 0;
    }

    void extend(size_t ExtraSize) {
        size_t newBlockOffset = m_TotalSize;
        size_t newBlockSize   = ExtraSize;

        if (!m_FreeBlocksByOffset.empty()) {
            auto lastBlockIt = m_FreeBlocksByOffset.end();
            --lastBlockIt;

            const auto lastBlockOffset = lastBlockIt->first;
            const auto lastBlockSize   = lastBlockIt->second.size;
            if (lastBlockOffset + lastBlockSize == m_TotalSize) {
                // Extend the last block
                newBlockOffset = lastBlockOffset;
                newBlockSize += lastBlockSize;

                FEYNMAN_ASSERT(lastBlockIt->second.orderBySizeIt->first == lastBlockSize &&
                            lastBlockIt->second.orderBySizeIt->second == lastBlockIt);
                m_FreeBlocksBySize.erase(lastBlockIt->second.orderBySizeIt);
                m_FreeBlocksByOffset.erase(lastBlockIt);
            }
        }

        addNewBlock(newBlockOffset, newBlockSize);

        m_TotalSize += ExtraSize;
        m_FreeSize += ExtraSize;
        FEYNMAN_ASSERT(m_FreeBlocksByOffset.size() == m_FreeBlocksBySize.size());
    }

    void debugPrint() {
        printf("********** free blocks **************\n");
        
        for (auto& [k, v] : m_FreeBlocksByOffset) {
            printf("unanlignedOffset:%llu size:%llu\n", k, v.size);
        }

        printf("********** busy blocks **************\n");

        std::vector<OffsetType> keys(m_BusyBlocksByAlignedOffset.size());

        auto key_selector = [](auto pair){return pair.first;};
        std::transform(m_BusyBlocksByAlignedOffset.begin(), m_BusyBlocksByAlignedOffset.end(), keys.begin(), key_selector);
        std::sort(keys.begin(), keys.end());

        for (auto& k : keys) {
            auto& v = m_BusyBlocksByAlignedOffset[k];
            printf("unanlignedOffset:%llu size:%llu\n", v.unalignedOffset, v.size);
        }
    }

private:

    /// allocate block with head padding
    ///  
    /// |<------------------result----------------->|
    /// |                   |<---- alignedSize ---->|
    /// |                   alignedOffset           |
    UnalignedAllocation allocateUnaligned(SizeType size, OffsetType alignment) {

        // check
        FEYNMAN_ASSERT(size > 0);
        FEYNMAN_ASSERT(Utils::isPowerOf2(alignment), "alignment {} must be power of 2", alignment);
        FEYNMAN_ASSERT(alignment >= m_MinAlignment, "alignment {} must >= m_MinAlignment {}", alignment, m_MinAlignment);

        // adjust size to alignment
        size = Utils::alignTo(size, alignment);
        if (m_FreeSize < size) {
            // FEYNMAN_THROW("allocate fail");
            return UnalignedAllocation::InvalidAllocation();
        }

        // find best fit block with head padding because of alignment
        auto alignmentHeadPadding = (alignment > m_CurrAlignment) ? alignment - m_CurrAlignment : 0;
        // Get the first block that is large enough to encompass size + alignmentHeadPadding bytes
        // lower_bound() returns an iterator pointing to the first element that
        // is not less (i.e. >= ) than key
        auto smallestBlockItIt = m_FreeBlocksBySize.lower_bound(size + alignmentHeadPadding);
        if (smallestBlockItIt == m_FreeBlocksBySize.end()) {
            // FEYNMAN_THROW("allocate fail");
            return UnalignedAllocation::InvalidAllocation();
        }

        // blockitit{k:size, v:blockit} -> blockit{k:offset, v:block}
        auto smallestBlockIt = smallestBlockItIt->second;
        OffsetType offset = smallestBlockIt->first;
        FreeBlockInfo& block = smallestBlockIt->second;
        
        FEYNMAN_ASSERT(size + alignmentHeadPadding <= block.size);
        FEYNMAN_ASSERT(block.size == smallestBlockItIt->first);

        //       smallestBlockIt.offset
        //          |                                               |
        //          |<---------------smallestBlockIt.size---------->|
        //          |<-------adjustedSize------>|<------newSize---->|
        //          |       |<------size ------>|                   |
        //          |       |                   |                   |
        //          |       alignedOffset       |                   |
        //          |                           |                   |
        //          |<----------result--------->|<-----residual---->|
        //          |                           |       
        //        offset                    newOffset
        //                       
        
        // adjust offset and size according above figure
        FEYNMAN_ASSERT(offset % m_CurrAlignment == 0, "");
        auto alignedOffset = Utils::alignTo(offset, alignment);
        auto adjustedSize  = size + (alignedOffset - offset);   // size + padding
        FEYNMAN_ASSERT(adjustedSize <= size + alignmentHeadPadding);
        auto newOffset = offset + adjustedSize;                 // next offset
        auto newSize   = smallestBlockIt->second.size - adjustedSize;
        FEYNMAN_ASSERT(smallestBlockItIt == smallestBlockIt->second.orderBySizeIt);

        // erase old block
        m_FreeBlocksBySize.erase(smallestBlockItIt);
        m_FreeBlocksByOffset.erase(smallestBlockIt);

        // add new residual block
        if (newSize > 0) {
            addNewBlock(newOffset, newSize);
        }

        m_FreeSize -= adjustedSize;

        // update m_CurrAlignment only if size < m_CurrAlignment, which left smaller alignment
        // size not multiple of m_CurrAlignment, i.e. has lower bit covered by alignment
        if ((size & (m_CurrAlignment - 1)) != 0) {
            if (Utils::isPowerOf2(size)) {  // only one bit is 1, size is smaller
                FEYNMAN_ASSERT(size >= alignment && size < m_CurrAlignment);
                m_CurrAlignment = size;
            } else {                        // size is bigger
                m_CurrAlignment = (std::min)(m_CurrAlignment, alignment);
            }
        }
        return UnalignedAllocation{offset, adjustedSize};
    }

    void free(OffsetType alignedOffset) {
        // find block
        if (auto it = m_BusyBlocksByAlignedOffset.find(alignedOffset); it != m_BusyBlocksByAlignedOffset.end()) {

            // remove busy block
            UnalignedAllocation unalignedAllocation = it->second;
            m_BusyBlocksByAlignedOffset.erase(alignedOffset);

            // free block
            freeUnaligned(std::move(unalignedAllocation));
        } else {
            FEYNMAN_THROW("free a space that never allocated");
        }
    }

    void freeUnaligned(UnalignedAllocation&& allocation) {
        FEYNMAN_ASSERT(allocation.isValid());
        freeUnaligned(allocation.unalignedOffset, allocation.size);
        allocation = UnalignedAllocation{};
    }

    void freeUnaligned(OffsetType offset, SizeType size) {

        FEYNMAN_ASSERT(offset != UnalignedAllocation::npos && offset + size <= m_TotalSize);

        // Find the first element whose offset is greater than the specified offset.
        // upper_bound() returns an iterator pointing to the first element in the
        // container whose key is considered to go after k.
        auto nextBlockIt = m_FreeBlocksByOffset.upper_bound(offset);
        // Block being deallocated must not overlap with the next block
        FEYNMAN_ASSERT(nextBlockIt == m_FreeBlocksByOffset.end() || offset + size <= nextBlockIt->first);
        auto prevBlockIt = nextBlockIt;
        if (prevBlockIt != m_FreeBlocksByOffset.begin()) {
            --prevBlockIt;
            // Block being deallocated must not overlap with the previous block
            FEYNMAN_ASSERT(offset >= prevBlockIt->first + prevBlockIt->second.size);
        } else {
            prevBlockIt = m_FreeBlocksByOffset.end();
        }

        OffsetType newSize, newOffset;
        if (prevBlockIt != m_FreeBlocksByOffset.end() && offset == prevBlockIt->first + prevBlockIt->second.size) {
            // can merge with previous
            
            //  PrevBlock.offset             offset
            //       |                          |
            //       |<-----PrevBlock.size----->|<------size-------->|
            //
            newSize   = prevBlockIt->second.size + size;
            newOffset = prevBlockIt->first;

            if (nextBlockIt != m_FreeBlocksByOffset.end() && offset + size == nextBlockIt->first) {
                // merge with both

                //   PrevBlock.offset           offset            NextBlock.offset
                //     |                          |                    |
                //     |<-----PrevBlock.size----->|<------size-------->|<-----NextBlock.size----->|
                //
                newSize += nextBlockIt->second.size;
                m_FreeBlocksBySize.erase(prevBlockIt->second.orderBySizeIt);
                m_FreeBlocksBySize.erase(nextBlockIt->second.orderBySizeIt);
                // Delete the range of two blocks
                ++nextBlockIt;
                m_FreeBlocksByOffset.erase(prevBlockIt, nextBlockIt);
            } else {
                // merge with previous only

                //   PrevBlock.offset           offset                     NextBlock.offset
                //     |                          |                             |
                //     |<-----PrevBlock.size----->|<------size-------->| ~ ~ ~  |<-----NextBlock.size----->|
                //
                m_FreeBlocksBySize.erase(prevBlockIt->second.orderBySizeIt);
                m_FreeBlocksByOffset.erase(prevBlockIt);
            }
        } else if (nextBlockIt != m_FreeBlocksByOffset.end() && offset + size == nextBlockIt->first) {
            // merge with next only

            //   PrevBlock.offset                   offset            NextBlock.offset
            //     |                                  |                    |
            //     |<-----PrevBlock.size----->| ~ ~ ~ |<------size-------->|<-----NextBlock.size----->|
            //
            newSize   = size + nextBlockIt->second.size;
            newOffset = offset;
            m_FreeBlocksBySize.erase(nextBlockIt->second.orderBySizeIt);
            m_FreeBlocksByOffset.erase(nextBlockIt);
        } else {
            // merge with none

            //   PrevBlock.offset                   offset                     NextBlock.offset
            //     |                                  |                            |
            //     |<-----PrevBlock.size----->| ~ ~ ~ |<------size-------->| ~ ~ ~ |<-----NextBlock.size----->|
            //
            newSize   = size;
            newOffset = offset;
        }

        addNewBlock(newOffset, newSize);

        m_FreeSize += size;
        if (isEmpty()) {
            // Reset current alignment
            FEYNMAN_ASSERT(getNumFreeBlocks() == 1);
            resetCurrAlignment();
        }
    }

    void addNewBlock(OffsetType offset, OffsetType size) {
        auto newBlockIt = m_FreeBlocksByOffset.emplace(offset, size);
        FEYNMAN_ASSERT(newBlockIt.second);

        auto OrderIt = m_FreeBlocksBySize.emplace(size, newBlockIt.first);
        newBlockIt.first->second.orderBySizeIt = OrderIt;
    }
    void resetCurrAlignment() {
        for (m_CurrAlignment = 1; m_CurrAlignment * 2 <= m_TotalSize; m_CurrAlignment *= 2){
        }
    }

private:
    TFreeBlocksByOffsetMap m_FreeBlocksByOffset;
    TFreeBlocksBySizeMap   m_FreeBlocksBySize;
    std::unordered_map<OffsetType, UnalignedAllocation> m_BusyBlocksByAlignedOffset;

    OffsetType m_TotalSize      = 0;    // total size
    OffsetType m_FreeSize       = 0;    // free size
    OffsetType m_CurrAlignment  = 0;    // min address alignment, init as max poweroftwo number that <= totalSize
    OffsetType m_MinAlignment   = 1;    // min alignment of allocate paramter

};


/// scenario: allocate shared srv for entire scene on hardware tier 2/3 who has unlimted srv per shader
/// 
/// This algorithm allocate one one by one with continous space without hole and without alignment
/// allocate one by one from old to new, release one by one from new to old, like stack
///
class StackAllocator : public IAllocator {
public:
    // ------------------------- type define
    using Ptr = std::shared_ptr<StackAllocator>;
    using OffsetType = size_t;
    using SizeType = size_t;

private:

public:

    explicit StackAllocator(SizeType totalSize): m_TotalSize(totalSize), m_FreeSize(totalSize) {
        
    }
    virtual ~StackAllocator() {
    }

    // ---------------------- adapter
    Allocation push(SizeType size) {
        return allocate(size, 1);
    }

    void pop() {
        FEYNMAN_ASSERT(m_BusyBlocks.size() > 0, "no block to free");
        auto block = m_BusyBlocks.back();
        free(block);
    }

    // ---------------------- interface implement
    /// allocate with exact size and no alignment
    virtual Allocation allocate(SizeType size, OffsetType alignment) override {
        FEYNMAN_ASSERT(alignment == 1, "StackAllocator has no alignment");

        if (m_FreeSize < size) {
            FEYNMAN_THROW("allocate fail");
            return Allocation::InvalidAllocation();
        }

        // allocate new
        Allocation allocation{m_CurrentOffset, size};
        // push allocation
        m_CurrentOffset += size;
        m_FreeSize -= size;
        m_BusyBlocks.push_back(allocation);
        return allocation;
    }
    virtual void free(Allocation& allocation) override {
        FEYNMAN_ASSERT(m_BusyBlocks.size() > 0, "no block to free");
        FEYNMAN_ASSERT(allocation == m_BusyBlocks.back(), "not last allocated block");

        // pop allocation
        m_FreeSize += m_BusyBlocks.back().size;
        m_BusyBlocks.pop_back();
        m_CurrentOffset = allocation.offset;
    }

    OffsetType getMaxSize() const{return m_TotalSize;}
    OffsetType getFreeSize()const override {return m_FreeSize;}
    OffsetType getUsedSize()const{return m_TotalSize - m_FreeSize;}

    void extend(size_t ExtraSize) {
        m_TotalSize += ExtraSize;
        m_FreeSize += ExtraSize;
    }

    void debugPrint() {
        printf("********** busy blocks **************\n");
        
        for (auto& v : m_BusyBlocks) {
            printf("offset:%llu size:%llu\n", v.offset, v.size);
        }
    }

private:
    OffsetType m_CurrentOffset  = 0;
    std::vector<Allocation> m_BusyBlocks;

    OffsetType m_TotalSize      = 0;    // total size
    OffsetType m_FreeSize       = 0;    // free size

};
/// scenario: allocate dynamic per shader srv
/// 
/// allocate chunks at unlimited space
/// inter chunk, try last chunk or all chunk, if fail, create new chunk
/// intra chunk, use ChunkAllocatorType allocator
///
/// The class facilitates allocation of dynamic descriptor handles. It requests a chunk of heap
/// from the master GPU descriptor heap and then performs linear suballocation within the chunk
/// At the end of the frame all allocations are disposed.
///
///     static and mutable handles     ||                 dynamic space
///                                    ||    chunk 0                 chunk 2
///  |                                 ||  | X X X O |             | O O O O |           || GPU Descriptor Heap
///                                        |                       |
///                                        m_Suballocations[0]     m_Suballocations[1]
///
template<typename IntraChunkAllocatorType, bool OnlyCheckLastChunk = true, typename ChunkCreateCallbackType = std::function<void(size_t, size_t)>, typename = std::enable_if_t<std::is_base_of_v<IAllocator, IntraChunkAllocatorType>>>
class ChunkAllocator : public IAllocator {
public:
    using Ptr = std::shared_ptr<ChunkAllocator>;
    using OffsetType = size_t;
    using SizeType = size_t;
private:
    class Chunk {
    public:
        Chunk(const UnalignedAllocation& allocation) : m_ChunkRange(allocation) {
            if constexpr (std::is_same_v<IntraChunkAllocatorType, StackAllocator>) {
                m_ChunkAllocator = std::make_shared<StackAllocator>(allocation.size);
            } else if constexpr (std::is_same_v<IntraChunkAllocatorType, BestFitAllocator>) {
                m_ChunkAllocator = std::make_shared<BestFitAllocator>(allocation.size);
            }
        }
        Allocation allocate(SizeType size, OffsetType alignment) {
            Allocation allocation = m_ChunkAllocator->allocate(size, alignment);
            // convert to global range
            if(allocation.offset != Allocation::npos) {
                allocation.offset += m_ChunkRange.unalignedOffset;
            }
            return allocation;
        }
        void free(Allocation& allocation) {
            // convert to local range
            allocation.offset -= m_ChunkRange.unalignedOffset;
            m_ChunkAllocator->free(allocation);
        }

        SizeType getFreeSize() const { return m_ChunkAllocator->getFreeSize(); }
        OffsetType getOffset() const { return m_ChunkRange.unalignedOffset; }
        SizeType getSize() const { return m_ChunkRange.size; }
        const UnalignedAllocation& getRange() const { return m_ChunkRange; }

        typename IntraChunkAllocatorType::Ptr getChunkAllocator() { return m_ChunkAllocator; }
        
    private:
        UnalignedAllocation m_ChunkRange;
        typename IntraChunkAllocatorType::Ptr m_ChunkAllocator;
    };

public:

    ChunkAllocator(SizeType totalSize = SizeType(-1), SizeType chunkSize = 16, ChunkCreateCallbackType callback = nullptr) 
    : mTotalSize(totalSize), mFreeSize(totalSize), mChunkSize(chunkSize), mChunkCreateCallback(callback) {
    }
    virtual ~ChunkAllocator() {
    }

    /// alignment must smaller than chunksize
    virtual Allocation allocate(SizeType size, OffsetType alignment) override {

        // make sure chunk header is alignment
        FEYNMAN_ASSERT(alignment < mChunkSize);

        // try to allocation on old chunks
        if (mChunks.size() != 0) {
            size_t startIndex = OnlyCheckLastChunk ? mChunks.size() - 1 : 0;
            for (size_t i = startIndex; i < mChunks.size(); i++) {
                size_t sizeBefore = mChunks[i].getFreeSize();
                Allocation allocation = mChunks[i].allocate(size, alignment);
                if (allocation.offset != Allocation::npos) {
                    size_t sizeAfter = mChunks[i].getFreeSize();
                    mFreeSize -= sizeBefore - sizeAfter;
                    return allocation;
                }
            }
        }

        // create new chunk
        SizeType alignedSize = Utils::alignTo(size, alignment);
        const UnalignedAllocation oldrange = mChunks.size() != 0 ? mChunks.back().getRange() : UnalignedAllocation{0, 0};
        UnalignedAllocation newrange = {oldrange.unalignedOffset + oldrange.size, mChunkSize};
        if (newrange.size < alignedSize) {
            newrange.size = alignedSize;
        }
        if (newrange.unalignedOffset + newrange.size >= mTotalSize) {
            return Allocation::InvalidAllocation();
        }
        mOffsetToChunkIndex[newrange.unalignedOffset] = mChunks.size();
        mChunks.push_back(Chunk(newrange));
        
        // callback
        if(mChunkCreateCallback != nullptr) {
            mChunkCreateCallback(newrange.unalignedOffset, newrange.size);
        }

        // allocate
        size_t sizeBefore = mChunks.back().getFreeSize();
        Allocation allocation = mChunks.back().allocate(size, alignment);
        size_t sizeAfter = mChunks.back().getFreeSize();
        mFreeSize -= sizeBefore - sizeAfter;
        return allocation;
    }
    virtual void free(Allocation& allocation) override {
        // find chunk
        auto index = getChunkIndex(allocation);

        // free
        size_t sizeBefore = mChunks[index].getFreeSize();
        mChunks[index].free(allocation);
        size_t sizeAfter = mChunks[index].getFreeSize();
        mFreeSize += sizeAfter - sizeBefore;
    }

    OffsetType getFreeSize()const override {return mFreeSize;}

    size_t getChunkIndex(const Allocation& allocation) const {
        FEYNMAN_ASSERT(mChunks.size() != 0);

        if (auto it = mOffsetToChunkIndex.upper_bound(allocation.offset); it != mOffsetToChunkIndex.end()) {
            // previous of it, it must not be first one
            it--;
            return it->second;
        } else {
            // previous of end
            return mChunks.size() - 1;
        }
    }

    void debugPrint() {
        printf("********** chunks begin **************\n");
        
        for (size_t i = 0; i < mChunks.size(); i++) {
            printf("********** chunk i:%d start:%llu size:%llu**************\n", i, mChunks[i].getOffset(),  mChunks[i].getSize());
            mChunks[i].getChunkAllocator()->debugPrint();
        }

        printf("********** chunks end **************\n");
    }    

private:
    SizeType mChunkSize;
    ChunkCreateCallbackType mChunkCreateCallback;

    SizeType mTotalSize;
    OffsetType mFreeSize = 0;
    std::vector<Chunk> mChunks;
    std::map<OffsetType, size_t> mOffsetToChunkIndex;
};


using StaticChunkAllocator = ChunkAllocator<BestFitAllocator, true>;
using DynamicChunkAllocator = ChunkAllocator<BestFitAllocator, false>;

/// allocate in surange and return the global range
/// alignment must less than subrange head alignment
class SubAllocator : public IAllocator {
public:
    // ------------------------- type define
    using Ptr = std::shared_ptr<BestFitAllocator>;
    using OffsetType = size_t;
    using SizeType = size_t;

public:
    SubAllocator(IAllocator::Ptr allocator, Allocation range, OffsetType headAlignment = 1)
        : mpAllocator(allocator), mRange(range), mHeadAlignment(headAlignment), mTotalSize(range.size), mFreeSize(range.size) {
        
        FEYNMAN_ASSERT(mpAllocator != nullptr, "");
    }

    virtual Allocation allocate(SizeType size, OffsetType alignment) override {

        FEYNMAN_ASSERT(alignment < mHeadAlignment, "");

        size_t sizeBefore = mpAllocator->getFreeSize();
        Allocation allocation = mpAllocator->allocate(size, alignment);
        size_t sizeAfter = mpAllocator->getFreeSize();

        // convert to global range
        if (allocation.offset != Allocation::npos) {
            mFreeSize -= sizeBefore - sizeAfter;
            mRange.offset += allocation.offset;
        }
        return allocation;
    }
    virtual void free(Allocation& allocation) override {
        FEYNMAN_ASSERT(allocation.isValid());

        // convert to local range
        allocation.offset -= mRange.offset;
        size_t sizeBefore = mpAllocator->getFreeSize();
        mpAllocator->free(allocation);
        size_t sizeAfter = mpAllocator->getFreeSize();
        allocation = Allocation{};

        mFreeSize += sizeAfter - sizeBefore;
    }

    virtual size_t getFreeSize() const override {
        return mFreeSize;
    }

private:
    IAllocator* mpParentAllocator;

    Allocation mRange;
    OffsetType mHeadAlignment = 1;
    IAllocator::Ptr mpAllocator;

    // stat
    SizeType mTotalSize;
    OffsetType mFreeSize = 0;
};

} // namespace feynman
